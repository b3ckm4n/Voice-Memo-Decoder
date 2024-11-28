from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
import requests
import openai
import os
import logging
import sys
import psutil
import time
import json
from werkzeug.utils import secure_filename
from pathlib import Path
from functools import wraps
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log all uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

class Config:
    UPLOAD_FOLDER = Path('uploads')
    TRANSCRIPTION_FOLDER = Path('transcriptions')
    ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'}
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB max file size

app = Flask(__name__)
app.config.from_object(Config)

# Ensure required directories exist
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
Config.TRANSCRIPTION_FOLDER.mkdir(exist_ok=True)

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    return wrapped

# Environment variable validation
def validate_env_vars():
    required_vars = ['WHISPER_API_KEY', 'OPENAI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

try:
    validate_env_vars()
    WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
except EnvironmentError as e:
    logger.error(f"Environment validation failed: {e}")
    raise

def get_system_metrics():
    """Get current system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'memory_available': psutil.virtual_memory().available / (1024 * 1024),  # MB
        'timestamp': time.time()
    }

class TranscriptionMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {'system': get_system_metrics()}

    def record_step(self, step_name, **kwargs):
        self.metrics[step_name] = {
            'timestamp': time.time() - self.start_time,
            'system_metrics': get_system_metrics(),
            **kwargs
        }

    def get_report(self):
        return {
            'total_duration': time.time() - self.start_time,
            'steps': self.metrics
        }

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_AUDIO_EXTENSIONS

def chunk_long_audio(file_path, max_chunk_size_mb=10):
    """Split audio into smaller chunks"""
    try:
        logger.info(f"Starting audio chunking for {file_path}")
        audio = AudioSegment.from_file(file_path)
        audio_length = len(audio)  # Duration in milliseconds

        logger.info(f"Audio length: {audio_length}ms")
        chunks = []

        # Calculate chunk size based on file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        chunk_duration = int((10 * 60 * 1000))  # 10 minutes in milliseconds

        if audio_length <= chunk_duration:
            logger.info("Audio file small enough to process as single chunk")
            return [file_path]

        # Split into chunks
        current_position = 0
        chunk_number = 0

        while current_position < audio_length:
            chunk_number += 1
            end_position = min(current_position + chunk_duration, audio_length)

            chunk = audio[current_position:end_position]

            # Export chunk with reduced quality
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                logger.info(f"Exporting chunk {chunk_number} to {temp_file.name}")
                chunk = chunk.set_channels(1)  # Convert to mono
                chunk = chunk.set_frame_rate(16000)  # Reduce sample rate

                chunk.export(
                    temp_file.name,
                    format='wav',
                    parameters=[
                        "-ar", "16000",  # Sample rate
                        "-ac", "1",      # Mono
                        "-b:a", "64k"    # Reduced bitrate
                    ]
                )

                # Check chunk size
                chunk_size_mb = os.path.getsize(temp_file.name) / (1024 * 1024)
                logger.info(f"Chunk {chunk_number} size: {chunk_size_mb:.2f}MB")

                # Reduce quality further if needed
                if chunk_size_mb > max_chunk_size_mb:
                    logger.warning(f"Chunk {chunk_number} is too large, reducing quality")
                    chunk = chunk.set_frame_rate(8000)
                    chunk.export(
                        temp_file.name,
                        format='wav',
                        parameters=[
                            "-ar", "8000",
                            "-ac", "1",
                            "-b:a", "32k"
                        ]
                    )

                chunks.append(temp_file.name)

            current_position = end_position

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Audio chunking failed: {e}", exc_info=True)
        raise

def transcribe_chunk(file_path):
    """Transcribe a single audio chunk"""
    try:
        logger.info(f"Starting transcription for chunk: {file_path}")
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Chunk file size: {file_size_mb:.2f}MB")

        if file_size_mb > 25:  # Whisper's limit
            logger.error(f"Chunk too large: {file_size_mb:.2f}MB")
            return {"error": "Chunk size exceeds API limit"}

        with open(file_path, 'rb') as audio_file:
            logger.info("Sending request to Whisper API")
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {WHISPER_API_KEY}"
                },
                files={
                    "file": (Path(file_path).name, audio_file, 'audio/wav')
                },
                data={
                    "model": "whisper-1",
                    "response_format": "text"  # Simplified response format
                },
                timeout=300
            )

            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code == 413:
                logger.error("Request too large (413). Attempting to reduce file size...")
                # If we get a 413, try to reduce the file size
                audio = AudioSegment.from_file(file_path)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    audio = audio.set_channels(1).set_frame_rate(8000)
                    audio.export(
                        temp_file.name,
                        format='wav',
                        parameters=["-ar", "8000", "-ac", "1", "-b:a", "32k"]
                    )
                    # Retry with reduced file
                    return transcribe_chunk(temp_file.name)

            if response.status_code != 200:
                error_content = response.content.decode('utf-8', errors='replace')
                logger.error(f"API Error Response: {error_content}")
                return {"error": f"API Error: {response.status_code}", "details": error_content}

            # For text response format
            transcription_text = response.text
            logger.info(f"Transcription successful: {len(transcription_text)} characters")
            return {"text": transcription_text}

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.route('/')
@error_handler
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@error_handler
def upload():
    metrics = TranscriptionMetrics()
    metrics.record_step('upload_start')
    logger.info("Starting new upload request")

    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file or not file.filename:
            logger.error("No file selected")
            return jsonify({"error": "No selected file"}), 400

        # Log original filename
        logger.info(f"Original filename: {file.filename}")

        # Clean filename more thoroughly
        original_filename = file.filename
        filename = secure_filename(original_filename)
        # Remove any remaining problematic characters
        filename = ''.join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))

        logger.info(f"Sanitized filename: {filename}")

        if not filename:
            logger.error("Filename became empty after sanitization")
            return jsonify({"error": "Invalid filename"}), 400

        if not allowed_file(filename):
            logger.error(f"File type not allowed: {filename}")
            return jsonify({"error": "File type not allowed"}), 400

        file_path = Config.UPLOAD_FOLDER / filename

        try:
            logger.info(f"Attempting to save file to: {file_path}")
            file.save(str(file_path))  # Convert Path to string
            logger.info(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")
        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            return jsonify({"error": f"Failed to save file: {str(save_error)}"}), 500

        try:
            metrics.record_step('transcription_start')
            transcription_response = transcribe_chunk(file_path)
            metrics.record_step('transcription_complete')

            if "error" in transcription_response:
                logger.error(f"Transcription error: {transcription_response['error']}")
                return jsonify({"error": transcription_response["error"]}), 500

            transcription_text = transcription_response.get('text', '')
            logger.info(f"Transcription completed successfully. Length: {len(transcription_text)} characters")

            # Save transcription
            transcription_file_path = Config.TRANSCRIPTION_FOLDER / "transcription.txt"
            with open(transcription_file_path, 'w', encoding='utf-8') as f:
                f.write(transcription_text)

            logger.info("Transcription saved successfully")
            return jsonify({
                "success": True,
                "redirect": url_for('transcription_preview')
            })

        except Exception as e:
            logger.exception("Transcription processing failed")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.exception("Upload processing failed")
        return jsonify({
            "error": str(e),
            "details": {
                "error_type": str(type(e)),
                "error_message": str(e)
            }
        }), 500

    finally:
        # Clean up uploaded file
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info("Cleaned up uploaded file")
        except Exception as e:
            logger.warning(f"Failed to clean up file: {e}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    try:
        # More thorough extension checking
        if '.' not in filename:
            logger.warning(f"No extension found in filename: {filename}")
            return False

        ext = filename.rsplit('.', 1)[1].lower()
        is_allowed = ext in Config.ALLOWED_AUDIO_EXTENSIONS

        logger.info(f"File extension check: {filename} -> {ext} -> {'allowed' if is_allowed else 'not allowed'}")
        return is_allowed
    except Exception as e:
        logger.error(f"Error checking file extension: {str(e)}")
        return False

@app.route('/transcription_preview')
@error_handler
def transcription_preview():
    try:
        transcription_path = Config.TRANSCRIPTION_FOLDER / "transcription.txt"
        transcription_text = transcription_path.read_text(encoding='utf-8')
        return render_template('transcription_preview.html', transcription_text=transcription_text)
    except FileNotFoundError:
        return jsonify({"error": "No transcription file found"}), 404

@app.route('/improve', methods=['POST'])
@error_handler
def improve():
    transcription_text = request.form.get('transcription_text')
    if not transcription_text:
        return jsonify({"error": "No transcription text provided"}), 400

    try:
        # Calculate approximate token count (rough estimation)
        token_estimate = len(transcription_text.split()) * 1.3  # Average words to tokens ratio
        logger.info(f"Estimated token count: {token_estimate}")

        # Split text into smaller chunks if needed (around 6000 tokens per chunk to be safe)
        max_chunk_words = 4000  # Approximately 5200 tokens
        words = transcription_text.split()
        chunks = []

        for i in range(0, len(words), max_chunk_words):
            chunk = ' '.join(words[i:i + max_chunk_words])
            chunks.append(chunk)

        logger.info(f"Split transcription into {len(chunks)} chunks")

        client = openai.OpenAI()
        improved_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "Clean up transcriptions by: 1) Fix grammar/punctuation 2) Add paragraphs 3) Label speakers as 'Speaker 1/2/etc' 4) Mark unclear parts as '[inaudible]'"
                        },
                        {"role": "user", "content": f"Clean up this part of the transcription:\n\n{chunk}"}
                    ],
                    temperature=0.3
                )

                improved_chunk = response.choices[0].message.content.strip()
                improved_chunks.append(improved_chunk)
                logger.info(f"Successfully processed chunk {i+1} of {len(chunks)}")

            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i+1}: {str(chunk_error)}")
                logger.error(f"Chunk content preview: {chunk[:200]}...")
                raise

        # Combine improved chunks
        improved_text = '\n\n'.join(improved_chunks)

        # Log improvement details
        logger.info(f"""
        Transcription improvement completed:
        Original length: {len(transcription_text)}
        Improved length: {len(improved_text)}
        Number of chunks: {len(chunks)}
        """)

        # Save improved transcription
        improved_path = Config.TRANSCRIPTION_FOLDER / "improved_transcription.txt"
        with open(improved_path, 'w', encoding='utf-8') as f:
            f.write(improved_text)

        # Log file save success
        logger.info(f"Saved improved transcription to {improved_path}")

        return render_template('transcription_result.html', 
                             improved_transcription=improved_text,
                             original_text=transcription_text)

    except openai.RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error: {str(e)}")
        return jsonify({"error": "Rate limit exceeded. Please try again in a few minutes."}), 429
    except openai.APIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return jsonify({"error": "API service error. Please try again later."}), 503
    except Exception as e:
        logger.error(f"Improvement failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Original text length: {len(transcription_text)}")
        return jsonify({
            "error": f"Failed to improve transcription: {str(e)}",
            "details": {
                "error_type": str(type(e)),
                "text_length": len(transcription_text)
            }
        }), 500

@app.route('/download')
@error_handler
def download():
    """Download the original transcription"""
    try:
        transcription_path = Config.TRANSCRIPTION_FOLDER / "transcription.txt"
        if not transcription_path.exists():
            return jsonify({"error": "No transcription file available"}), 404

        return send_file(
            transcription_path,
            as_attachment=True,
            download_name='transcription.txt',
            mimetype='text/plain'
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({"error": "Failed to download transcription"}), 500

@app.route('/debug/status')
def debug_status():
    """Debug endpoint to check system status"""
    try:
        upload_dir = Config.UPLOAD_FOLDER
        transcription_dir = Config.TRANSCRIPTION_FOLDER

        return jsonify({
            "status": "running",
            "upload_directory": {
                "path": str(upload_dir),
                "exists": upload_dir.exists(),
                "is_dir": upload_dir.is_dir() if upload_dir.exists() else False,
                "permissions": oct(os.stat(upload_dir).st_mode)[-3:] if upload_dir.exists() else None
            },
            "transcription_directory": {
                "path": str(transcription_dir),
                "exists": transcription_dir.exists(),
                "is_dir": transcription_dir.is_dir() if transcription_dir.exists() else False,
                "permissions": oct(os.stat(transcription_dir).st_mode)[-3:] if transcription_dir.exists() else None
            },
            "environment": {
                "whisper_api_key_set": bool(os.getenv("WHISPER_API_KEY")),
                "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
                "python_version": sys.version,
            },
            "recent_logs": get_recent_logs()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

def get_recent_logs(num_lines=50):
    """Get the most recent log entries"""
    try:
        log_path = Path('transcription.log')
        if not log_path.exists():
            return ["No log file found"]

        with open(log_path, 'r') as f:
            lines = f.readlines()
            return lines[-num_lines:] if lines else ["Log file is empty"]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]

@app.route('/health')
def health_check():
    """Endpoint to check system health and resources"""
    return jsonify({
        "status": "healthy",
        "system_metrics": get_system_metrics(),
        "storage": {
            "uploads": str(Config.UPLOAD_FOLDER),
            "transcriptions": str(Config.TRANSCRIPTION_FOLDER),
            "disk_free": psutil.disk_usage('/').free / (1024 * 1024)  # MB
        }
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)