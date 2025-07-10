from flask import Flask, render_template, request, jsonify, session, Response
import sys
import pickle
from utils import *
from options import args
from models import model_factory
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
import random
import re
import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
import gdown

app = Flask(__name__)
app.secret_key = '1903bjk'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
chat_messages = []
active_users = {}
MAX_MESSAGES = 300
recommendation_system = None
system_status = {"status": "loading", "message": "Initializing system..."}

def generate_username():
    adjectives = ['Cool', 'Awesome', 'Swift', 'Bright', 'Happy', 'Smart', 'Kind', 'Brave', 'Calm', 'Epic', "Black"]
    nouns = ['Otaku', 'Ninja', 'Samurai', 'Dragon', 'Phoenix', 'Tiger', 'Wolf', 'Eagle', 'Fox', 'Bear']
    return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(100, 999)}"

def clean_message(message):
    message = re.sub(r'<[^>]*>', '', message)
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
        self.load_model_and_data()

    def load_model_and_data(self):
        try:
            print("Loading model and data...")
            global system_status
            system_status["message"] = "Loading datasets..."

            args.bert_max_len = 128

            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)
            
            system_status["message"] = "Loading anime data..."
            with open(self.animes_path, "r", encoding="utf-8") as file:
                self.id_to_anime = json.load(file)

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

            system_status["message"] = "Loading AI model..."
            self.model = model_factory(args)
            self.load_checkpoint()

            system_status["status"] = "ready"
            system_status["message"] = "System ready!"
            print("Model loaded successfully!")

        except Exception as e:
            system_status["status"] = "error"
            system_status["message"] = f"Error loading model: {str(e)}"
            print(f"Error loading model: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("Model checkpoint loaded successfully!")
        except Exception as e:
            print(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")
            raise Exception(f"Failed to load checkpoint: {str(e)}")

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

    def get_filtered_anime_pool(self, filters):
        """Filtrelere göre anime havuzunu önceden filtreler"""
        if not filters:
            return None

        if filters.get('show_hentai') and len([k for k, v in filters.items() if v]) == 1:
            hentai_animes = []
            for anime_id_str, anime_data in self.id_to_anime.items():
                anime_id = int(anime_id_str)
                if self._is_hentai(anime_id):
                    hentai_animes.append(anime_id)
            return hentai_animes

        return None

    def _is_hentai(self, anime_id):
        """Anime'nin hentai olup olmadığını kontrol eder"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 3:
            return False
        return type_seq_info[2]

    def _get_type(self, anime_id):
        """Anime'nin türünü döndürür"""
        type_seq_info = self.id_to_type_seq.get(str(anime_id))
        if not type_seq_info or len(type_seq_info) < 3:
            return "Unknown"
        return type_seq_info[1]

    def get_recommendations(self, favorite_anime_ids, num_recommendations=40, filters=None):
        try:
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

            filtered_pool = self.get_filtered_anime_pool(filters)
            if filtered_pool is not None:
                return self._get_recommendations_from_pool(favorite_anime_ids, filtered_pool, num_recommendations, filters)

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

    def _get_recommendations_from_pool(self, favorite_anime_ids, anime_pool, num_recommendations, filters):
        """Önceden filtrelenmiş anime havuzundan öneriler alır"""
        try:
            smap = self.dataset
            inverted_smap = {v: k for k, v in smap.items()}

            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"

            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]

            anime_scores = []
            for anime_id in anime_pool:
                if anime_id in favorite_anime_ids:
                    continue

                if anime_id in smap:
                    model_id = smap[anime_id]
                    if model_id < last_logits.shape[1]:
                        score = last_logits[0, model_id].item()
                        anime_scores.append((anime_id, score))

            anime_scores.sort(key=lambda x: x[1], reverse=True)

            recommendations = []
            for anime_id, score in anime_scores[:num_recommendations]:
                if str(anime_id) in self.id_to_anime:
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

            return recommendations, [r['score'] for r in recommendations], f"Found {len(recommendations)} filtered recommendations!"

        except Exception as e:
            return [], [], f"Error during filtered prediction: {str(e)}"

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
        is_hentai = type_seq_info[2]

        if 'show_sequels' in filters:
            if not filters['show_sequels'] and is_sequel:
                return False

        if 'show_hentai' in filters:
            if filters['show_hentai']:
                if not is_hentai:
                    return False
            else:
                if is_hentai:
                    return False

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

def download_and_initialize_system():
    """Sistem dosyalarını indir ve başlat"""
    global recommendation_system, system_status
    
    try:
        args.num_items = 12689
        
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
                system_status["message"] = f"Downloading {output_path}..."
                gdown.download(url, output_path, quiet=False)
                print(f"Downloaded: {output_path}")
                return True
            except Exception as e:
                print(f"Error downloading {output_path}: {e}")
                return False
        
        # Dosyaları kontrol et ve indir
        for file_id, filename in file_ids.items():
            if not os.path.exists(filename):
                print(f"File {filename} not found, downloading...")
                if not download_from_gdrive(file_id, filename):
                    system_status["status"] = "error"
                    system_status["message"] = f"Failed to download {filename}"
                    return
            else:
                print(f"File {filename} already exists")
        
        # Sistem başlat
        print("Initializing recommendation system...")
        system_status["message"] = "Initializing recommendation system..."
        
        recommendation_system = AnimeRecommendationSystem(
            "pretrained_bert.pth",
            "dataset.pkl", 
            "animes.json",
            "id_to_url.json",
            "anime_to_malurl.json",
            "anime_to_typenseq.json",
            "id_to_genres.json"
        )
        
        print("Recommendation system initialized successfully!")
        
    except Exception as e:
        system_status["status"] = "error"
        system_status["message"] = f"Failed to initialize: {str(e)}"
        print(f"Failed to initialize recommendation system: {e}")

# Sistem durumu endpoint'i
@app.route('/api/system_status')
def get_system_status():
    return jsonify(system_status)

# Ana sayfa
@app.route('/')
def index():
    if system_status["status"] == "loading":
        return render_template('loading.html', message=system_status["message"])
    elif system_status["status"] == "error":
        return render_template('error.html', error=system_status["message"])
    elif recommendation_system is None:
        return render_template('loading.html', message="System is still loading...")
    
    try:
        animes = recommendation_system.get_all_animes()
        return render_template('index.html', animes=animes)
    except Exception as e:
        return render_template('error.html', error=f"Error loading animes: {str(e)}")

# API endpoint'leri
@app.route('/api/search_animes')
def search_animes():
    if recommendation_system is None:
        return jsonify({'error': 'System not ready'}), 503
    
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
    if recommendation_system is None:
        return jsonify({'error': 'System not ready'}), 503
        
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
    if recommendation_system is None:
        return jsonify({'success': False, 'message': 'System not ready, please wait...'}), 503
        
    if 'favorites' not in session or not session['favorites']:
        return jsonify({'success': False, 'message': 'Please add some favorite animes first!'})

    data = request.get_json() or {}
    filters = data.get('filters', {})

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

# SEO route'ları
@app.route('/sitemap.xml')
def sitemap():
    """Dinamik sitemap.xml oluşturur"""
    try:
        urlset = ET.Element('urlset')
        urlset.set('xmlns', 'http://www.sitemaps.org/schemas/sitemap/0.9')
        urlset.set('xmlns:image', 'http://www.google.com/schemas/sitemap-image/1.1')

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

        if recommendation_system and recommendation_system.id_to_anime:
            anime_count = 0
            for anime_id, anime_data in recommendation_system.id_to_anime.items():
                if anime_count >= 100:
                    break

                try:
                    anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
                    safe_name = anime_name.replace(' ', '-').replace('/', '-').replace('?', '').replace('&', 'and')
                    safe_name = re.sub(r'[^\w\-]', '', safe_name)

                    url = ET.SubElement(urlset, 'url')
                    ET.SubElement(url, 'loc').text = f'{base_url}/anime/{anime_id}/{safe_name}'
                    ET.SubElement(url, 'lastmod').text = current_date
                    ET.SubElement(url, 'changefreq').text = 'weekly'
                    ET.SubElement(url, 'priority').text = '0.6'

                    image_url = recommendation_system.get_anime_image_url(int(anime_id))
                    if image_url:
                        image_elem = ET.SubElement(url, 'image:image')
                        ET.SubElement(image_elem, 'image:loc').text = image_url
                        ET.SubElement(image_elem, 'image:title').text = anime_name
                        ET.SubElement(image_elem, 'image:caption').text = f'Poster image for {anime_name}'

                    anime_count += 1
                except Exception as e:
                    print(f"Error processing anime {anime_id}: {e}")
                    continue

        xml_str = ET.tostring(urlset, encoding='unicode')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        full_xml = xml_declaration + xml_str

        return Response(full_xml, mimetype='application/xml')

    except Exception as e:
        print(f"Sitemap generation error: {e}")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>',
            mimetype='application/xml')

@app.route('/robots.txt')
def robots_txt():
    """Robots.txt dosyası"""
    robots_content = f"""User-agent: *
Allow: /
Allow: /chat

Sitemap: {request.url_root.rstrip('/')}/sitemap.xml
"""
    return Response(robots_content, mimetype='text/plain')

@app.route('/chat')
def chat():
    return render_template('chat.html')

# SocketIO event'leri
@socketio.on('connect')
def on_connect():
    username = generate_username()
    active_users[request.sid] = {
        'username': username,
        'connected_at': datetime.now()
    }

    emit('chat_history', chat_messages)

    join_message = {
        'username': 'System',
        'message': f'{username} joined the chat',
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'system'
    }

    chat_messages.append(join_message)
    if len(chat_messages) > MAX_MESSAGES:
        chat_messages.pop(0)

    emit('new_message', join_message, broadcast=True)
    emit('user_count', len(active_users), broadcast=True)

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in active_users:
        username = active_users[request.sid]['username']
        del active_users[request.sid]

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

    message_obj = {
        'username': username,
        'message': message,
        'timestamp': datetime.now().strftime('%H:%M'),
        'type': 'user'
    }

    chat_messages.append(message_obj)
    if len(chat_messages) > MAX_MESSAGES:
        chat_messages.pop(0)

    emit('new_message', message_obj, broadcast=True)

@socketio.on('change_username')
def handle_username_change(data):
    if request.sid not in active_users:
        return

    old_username = active_users[request.sid]['username']
    new_username = clean_message(data.get('username', ''))

    if not new_username or len(new_username) < 2:
        return

    active_users[request.sid]['username'] = new_username

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

if __name__ == "__main__":
    # Sistem başlatmayı arka planda yap
    initialization_thread = threading.Thread(target=download_and_initialize_system)
    initialization_thread.daemon = True
    initialization_thread.start()
    
    # Render.com için port ayarı
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

