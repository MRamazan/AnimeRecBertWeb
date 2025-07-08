from flask import Flask, render_template, request, jsonify, session, Response
import sys
import pickle
import json
import torch
from pathlib import Path
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
import random
import re
import os
import xml.etree.ElementTree as ET

# Import your custom modules - make sure these files exist
try:
    from utils import *
    from options import args
    from models import model_factory
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure utils.py, options.py, and models.py exist in your project directory")
    # Create minimal args object if options.py doesn't exist
    class Args:
        bert_max_len = 128
        num_items = 12689
    args = Args()

port = int(os.environ.get("PORT", 5000))

app = Flask(__name__)
app.secret_key = '1903bjk'
socketio = SocketIO(app, cors_allowed_origins="*")

chat_messages = []
active_users = {}
MAX_MESSAGES = 300

def generate_username():
    adjectives = ['Cool', 'Awesome', 'Swift', 'Bright', 'Happy', 'Smart', 'Kind', 'Brave', 'Calm', 'Epic', "Black"]
    nouns = ['Otaku', 'Ninja', 'Samurai', 'Dragon', 'Phoenix', 'Tiger', 'Wolf', 'Eagle', 'Fox', 'Bear']
    return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(100, 999)}"

def clean_message(message):
    # HTML tag'leri temizle
    message = re.sub(r'<[^>]*>', '', message)
    # Uzunluk kontrolü
    if len(message) > 500:
        message = message[:500]
    return message.strip()

class AnimeRecommendationSystem:
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path, mal_urls_path, type_seq_path, genres_path):
        self.model = None
        self.dataset = None
        self.id_to_anime = {}
        self.id_to_url = {}
        self.id_to_mal_url = {}
        self.genres_path = genres_path
        self.id_to_genres = {}
        self.type_seq_path = type_seq_path
        self.id_to_type_seq = {}
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.animes_path = animes_path
        self.images_path = images_path
        self.mal_urls_path = mal_urls_path
        
        # Initialize with error handling
        self.initialization_error = None
        try:
            self.load_model_and_data()
        except Exception as e:
            self.initialization_error = str(e)
            print(f"Failed to initialize recommendation system: {e}")

    def is_initialized(self):
        return self.initialization_error is None and self.model is not None

    def get_initialization_error(self):
        return self.initialization_error

    def load_model_and_data(self):
        try:
            print("Loading model and data...")

            # Check if required files exist
            required_files = [
                self.dataset_path,
                self.animes_path,
                self.checkpoint_path
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")

            # Set BERT max length
            if hasattr(args, 'bert_max_len'):
                args.bert_max_len = 128

            # Load dataset
            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)
            print(f"Loaded dataset with {len(self.dataset)} items")

            # Load anime data
            with open(self.animes_path, "r", encoding="utf-8") as file:
                self.id_to_anime = json.load(file)
            print(f"Loaded {len(self.id_to_anime)} anime entries")

            # Load optional files with error handling
            try:
                with open(self.images_path, "r", encoding="utf-8") as file:
                    self.id_to_url = json.load(file)
                print(f"Loaded {len(self.id_to_url)} image URLs")
            except Exception as e:
                print(f"Warning: Could not load image URLs: {str(e)}")
                self.id_to_url = {}

            try:
                with open(self.mal_urls_path, "r", encoding="utf-8") as file:
                    self.id_to_mal_url = json.load(file)
                print(f"Loaded {len(self.id_to_mal_url)} MAL URLs")
            except Exception as e:
                print(f"Warning: Could not load MAL URLs: {str(e)}")
                self.id_to_mal_url = {}

            try:
                with open(self.type_seq_path, "r", encoding="utf-8") as file:
                    self.id_to_type_seq = json.load(file)
                print(f"Loaded {len(self.id_to_type_seq)} type/sequel info")
            except Exception as e:
                print(f"Warning: Could not load type/sequel info: {str(e)}")
                self.id_to_type_seq = {}

            try:
                with open(self.genres_path, "r", encoding="utf-8") as file:
                    self.id_to_genres = json.load(file)
                print(f"Loaded {len(self.id_to_genres)} genre info")
            except Exception as e:
                print(f"Warning: Could not load genres: {str(e)}")
                self.id_to_genres = {}

            # Load model
            try:
                self.model = model_factory(args)
                self.load_checkpoint()
                print("Model loaded successfully!")
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")

        except Exception as e:
            print(f"Error loading model and data: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            
            # Check if PyTorch is available
            if not torch:
                raise ImportError("PyTorch is not installed")
            
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint does not contain 'model_state_dict'")
                
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Checkpoint loaded successfully!")
            
        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")

    def get_anime_genres(self, anime_id):
        genres = self.id_to_genres.get(str(anime_id), [])
        return [genre.title() for genre in genres] if genres else []

    def get_all_animes(self):
        """Tüm anime listesini döndürür"""
        animes = []
        for k, v in self.id_to_anime.items():
            anime_name = v[0] if isinstance(v, list) and len(v) > 0 else str(v)
            animes.append((int(k), anime_name))
        animes.sort(key=lambda x: x[1])
        return animes

    def get_anime_image_url(self, anime_id):
        return self.id_to_url.get(str(anime_id), None)

    def get_anime_mal_url(self, anime_id):
        return self.id_to_mal_url.get(str(anime_id), None)

    def _is_hentai(self, anime_id):
        """Anime'nin hentai olup olmadığını kontrol eder"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 3:
            return False
        return type_seq_info[2]

    def _get_type(self, anime_id):
        """Anime'nin tipini döndürür"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 2:
            return "Unknown"
        return type_seq_info[1]

    def get_recommendations(self, favorite_anime_ids, num_recommendations=40, filters=None):
        try:
            if not self.is_initialized():
                return [], [], f"Recommendation system not initialized: {self.get_initialization_error()}"

            if not favorite_anime_ids:
                return [], [], "Please add some favorite animes first!"

            smap = self.dataset
            inverted_smap = {v: k for k, v in smap.items()}

            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"

            # Normal öneriler
            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            max_predictions = min(500, len(inverted_smap))

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)

            recommendations = []
            scores = []

            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]

                    if anime_id in favorite_anime_ids:
                        continue

                    if str(anime_id) in self.id_to_anime:
                        # Filtreleme kontrolü
                        if filters and not self._should_include_anime(anime_id, filters):
                            continue

                        anime_data = self.id_to_anime[str(anime_id)]
                        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
                        image_url = self.get_anime_image_url(anime_id)
                        mal_url = self.get_anime_mal_url(anime_id)

                        recommendations.append({
                            'id': anime_id,
                            'name': anime_name,
                            'score': float(score),
                            'image_url': image_url,
                            'mal_url': mal_url,
                            'genres': self.get_anime_genres(anime_id)
                        })
                        scores.append(float(score))

                        if len(recommendations) >= num_recommendations:
                            break

            return recommendations, scores, f"Found {len(recommendations)} recommendations!"

        except Exception as e:
            return [], [], f"Error during prediction: {str(e)}"

    def _should_include_anime(self, anime_id, filters):
        """Filtrelere göre anime'nin dahil edilip edilmeyeceğini kontrol eder"""
        if 'blacklisted_animes' in filters:
            if anime_id in filters['blacklisted_animes']:
                return False

        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 2:
            return True

        anime_type = type_seq_info[0]
        is_sequel = type_seq_info[1]
        is_hentai = type_seq_info[2] if len(type_seq_info) > 2 else False

        # Sequel filtresi
        if 'show_sequels' in filters:
            if not filters['show_sequels'] and is_sequel:
                return False

        # Hentai filtresi
        if 'show_hentai' in filters:
            if filters['show_hentai']:
                if not is_hentai:
                    return False
            else:
                if is_hentai:
                    return False

        # Tür filtreleri
        if 'show_movies' in filters:
            if not filters['show_movies'] and anime_type == 'MOVIE':
                return False

        if 'show_tv' in filters:
            if not filters['show_tv'] and anime_type == 'TV':
                return False

        if 'show_ova' in filters:
            if not filters['show_ova'] and anime_type in ['ONA', 'OVA', 'SPECIAL']:
                return False

        return True

# Global variable for recommendation system
recommendation_system = None

def download_files():
    """Download required files from Google Drive"""
    try:
        import gdown
        
        file_ids = {
            "1C6mdjblhiWGhRgbIk5DP2XCc4ElS9x8p": "pretrained_bert.pth",
            "1U42cFrdLFT8NVNikT9C5SD9aAux7a5U2": "animes.json",
            "1s-8FM1Wi2wOWJ9cstvm-O1_6XculTcTG": "dataset.pkl",
            "1SOm1llcTKfhr-RTHC0dhaZ4AfWPs8wRx": "id_to_url.json",
            "1vwJEMEOIYwvCKCCbbeaP0U_9L3NhvBzg": "anime_to_malurl.json",
            "1_TyzON6ie2CqvzVNvPyc9prMTwLMefdu": "anime_to_typenseq.json",
            "1G9O_ahyuJ5aO0cwoVnIXrlzMqjKrf2aw": "id_to_genres.json"
        }

        def download_from_gdrive(file_id, output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                print(f"Downloading: {output_path}")
                gdown.download(url, output_path, quiet=False)
                print(f"Downloaded: {output_path}")
                return True
            except Exception as e:
                print(f"Error downloading {output_path}: {e}")
                return False

        for file_id, filename in file_ids.items():
            if not os.path.isfile(filename):
                print(f"File {filename} not found, attempting to download...")
                if not download_from_gdrive(file_id, filename):
                    print(f"Failed to download {filename}")
                    return False
            else:
                print(f"File {filename} already exists")
        
        return True
    except ImportError:
        print("gdown not installed. Install it with: pip install gdown")
        return False
    except Exception as e:
        print(f"Error in download_files: {e}")
        return False

def initialize_recommendation_system():
    """Initialize the recommendation system with error handling"""
    global recommendation_system
    
    try:
        # Set args
        if hasattr(args, 'num_items'):
            args.num_items = 12689

        # Define file paths
        images_path = "id_to_url.json"
        mal_urls_path = "anime_to_malurl.json"
        type_seq_path = "anime_to_typenseq.json"
        genres_path = "id_to_genres.json"

        # Check if required files exist
        required_files = ["pretrained_bert.pth", "dataset.pkl", "animes.json"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            print("Attempting to download files...")
            if not download_files():
                raise Exception("Failed to download required files")

        # Initialize recommendation system
        recommendation_system = AnimeRecommendationSystem(
            "pretrained_bert.pth",
            "dataset.pkl",
            "animes.json",
            images_path,
            mal_urls_path,
            type_seq_path,
            genres_path
        )
        
        if recommendation_system.is_initialized():
            print("Recommendation system initialized successfully!")
            return True
        else:
            print(f"Recommendation system failed to initialize: {recommendation_system.get_initialization_error()}")
            return False
            
    except Exception as e:
        print(f"Failed to initialize recommendation system: {e}")
        recommendation_system = None
        return False

# Routes
@app.route('/')
def index():
    if recommendation_system is None or not recommendation_system.is_initialized():
        error_msg = "Recommendation system not initialized. Please check server logs."
        if recommendation_system:
            error_msg += f" Error: {recommendation_system.get_initialization_error()}"
        return render_template('error.html', error=error_msg)

    animes = recommendation_system.get_all_animes()
    return render_template('index.html', animes=animes)

@app.route('/api/search_animes')
def search_animes():
    if recommendation_system is None or not recommendation_system.is_initialized():
        return jsonify({'error': 'Recommendation system not initialized'}), 500
        
    query = request.args.get('q', '').lower()
    animes = []

    for k, v in recommendation_system.id_to_anime.items():
        anime_names = v if isinstance(v, list) else [v]

        match_found = False
        for name in anime_names:
            if query in name.lower():
                match_found = True
                break

        if not query or match_found:
            main_name = anime_names[0] if anime_names else "Unknown"
            animes.append((int(k), main_name))

    animes.sort(key=lambda x: x[1])
    return jsonify(animes)

@app.route('/api/add_favorite', methods=['POST'])
def add_favorite():
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])
    anime_name = data['anime_name']

    if anime_id not in session['favorites']:
        session['favorites'].append(anime_id)
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/api/remove_favorite', methods=['POST'])
def remove_favorite():
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])

    if anime_id in session['favorites']:
        session['favorites'].remove(anime_id)
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/api/clear_favorites', methods=['POST'])
def clear_favorites():
    session['favorites'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/get_favorites')
def get_favorites():
    if recommendation_system is None or not recommendation_system.is_initialized():
        return jsonify({'error': 'Recommendation system not initialized'}), 500
        
    if 'favorites' not in session:
        session['favorites'] = []

    favorite_animes = []
    for anime_id in session['favorites']:
        if str(anime_id) in recommendation_system.id_to_anime:
            anime_data = recommendation_system.id_to_anime[str(anime_id)]
            anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
            favorite_animes.append({'id': anime_id, 'name': anime_name})

    return jsonify(favorite_animes)

@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    if recommendation_system is None or not recommendation_system.is_initialized():
        return jsonify({'success': False, 'message': 'Recommendation system not initialized'})
        
    if 'favorites' not in session or not session['favorites']:
        return jsonify({'success': False, 'message': 'Please add some favorite animes first!'})

    data = request.get_json() or {}
    filters = data.get('filters', {})

    # Blacklist bilgisini ekle
    blacklisted_animes = data.get('blacklisted_animes', [])
    if blacklisted_animes:
        filters['blacklisted_animes'] = blacklisted_animes

    recommendations, scores, message = recommendation_system.get_recommendations(
        session['favorites'],
        filters=filters
    )

    if recommendations:
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'message': message
        })
    else:
        return jsonify({'success': False, 'message': message})

@app.route('/api/mal_logo')
def get_mal_logo():
    return jsonify({
        'success': True,
        'logo_url': 'https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png'
    })

# Chat functionality
@app.route('/chat')
def chat():
    return render_template('chat.html')

# SocketIO events
@socketio.on('connect')
def on_connect():
    username = generate_username()
    active_users[request.sid] = {
        'username': username,
        'connected_at': datetime.now()
    }

    # Kullanıcıya mevcut mesajları gönder
    emit('chat_history', chat_messages)

    # Kullanıcı katıldı mesajı
    join_message = {
        'username': 'System',
        'message': f'{username} joined the chat',
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'system'
    }

    chat_messages.append(join_message)
    if len(chat_messages) > MAX_MESSAGES:
        chat_messages.pop(0)

    # Herkese gönder
    emit('new_message', join_message, broadcast=True)
    emit('user_count', len(active_users), broadcast=True)

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in active_users:
        username = active_users[request.sid]['username']
        del active_users[request.sid]

        # Kullanıcı ayrıldı mesajı
        leave_message = {
            'username': 'System',
            'message': f'{username} left the chat',
            'timestamp': datetime.now().strftime('%H:%M'),
            'type': 'system'
        }

        chat_messages.append(leave_message)
        if len(chat_messages) > MAX_MESSAGES:
            chat_messages.pop(0)

        emit('new_message', leave_message, broadcast=True)
        emit('user_count', len(active_users), broadcast=True)

@socketio.on('send_message')
def handle_message(data):
    if request.sid not in active_users:
        return

    username = active_users[request.sid]['username']
    message = clean_message(data.get('message', ''))

    if not message:
        return

    # Mesaj objesi oluştur
    message_obj = {
        'username': username,
        'message': message,
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'user'
    }

    # Mesajı kaydet
    chat_messages.append(message_obj)
    if len(chat_messages) > MAX_MESSAGES:
        chat_messages.pop(0)

    # Herkese gönder
    emit('new_message', message_obj, broadcast=True)

@socketio.on('change_username')
def handle_username_change(data):
    if request.sid not in active_users:
        return

    old_username = active_users[request.sid]['username']
    new_username = clean_message(data.get('username', ''))

    if not new_username or len(new_username) < 2:
        return

    # Kullanıcı adını güncelle
    active_users[request.sid]['username'] = new_username

    # İsim değişikliği mesajı
    change_message = {
        'username': 'System',
        'message': f'{old_username} changed name to {new_username}',
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'system'
    }

    chat_messages.append(change_message)
    if len(chat_messages) > MAX_MESSAGES:
        chat_messages.pop(0)

    emit('new_message', change_message, broadcast=True)
    emit('username_changed', {'username': new_username})

# SEO routes (simplified)
@app.route('/robots.txt')
def robots_txt():
    robots_content = f"""User-agent: *
Allow: /
Allow: /chat

Sitemap: {request.url_root.rstrip('/')}/sitemap.xml
"""
    return Response(robots_content, mimetype='text/plain')

@app.route('/sitemap.xml')
def sitemap():
    try:
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        
        base_url = request.url_root.rstrip('/')
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Ana sayfa
        url = ET.SubElement(urlset, 'url')
        ET.SubElement(url, 'loc').text = f'{base_url}/'
        ET.SubElement(url, 'lastmod').text = current_date
        ET.SubElement(url, 'changefreq').text = 'daily'
        ET.SubElement(url, 'priority').text = '1.0'

        # Chat sayfası
        url = ET.SubElement(urlset, 'url')
        ET.SubElement(url, 'loc').text = f'{base_url}/chat'
        ET.SubElement(url, 'lastmod').text = current_date
        ET.SubElement(url, 'changefreq').text = 'hourly'
        ET.SubElement(url, 'priority').text = '0.8'

        xml_str = ET.tostring(urlset, encoding='unicode')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        full_xml = xml_declaration + xml_str

        return Response(full_xml, mimetype='application/xml')
    except Exception as e:
        print(f"Sitemap generation error: {e}")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>',
            mimetype='application/xml')

def main():
    # Initialize recommendation system
    if not initialize_recommendation_system():
        print("Warning: Recommendation system initialization failed!")
        print("The server will start but recommendations won't work.")

    # Start the application
    socketio.run(app, host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main()
