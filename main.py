from flask import Flask, render_template, request, jsonify, session, Response
import sys
import pickle
import json
import os
import gzip
import weakref
import gc
from pathlib import Path
from utils import *
from options import args
from models import model_factory
from flask_socketio import SocketIO, emit
from datetime import datetime
import random
import re
import xml.etree.ElementTree as ET
import torch

app = Flask(__name__)
app.secret_key = '1903bjk'
socketio = SocketIO(app, cors_allowed_origins="*")

# RAM optimizasyonu için global değişkenler
chat_messages = []
active_users = {}
MAX_MESSAGES = 100  # Azaltıldı: 300 -> 100
MODEL_CACHE = {}
LAZY_LOAD_THRESHOLD = 1000  # Lazy loading için eşik

class MemoryOptimizedDict:
    """RAM dostu dictionary implementasyonu"""
    def __init__(self, data_path=None, max_cache_size=1000):
        self.data_path = data_path
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
        self._full_data = None
        
    def _load_full_data(self):
        """Tüm veriyi yükle (lazy loading)"""
        if self._full_data is None and self.data_path:
            try:
                if self.data_path.endswith('.gz'):
                    with gzip.open(self.data_path, 'rt', encoding='utf-8') as f:
                        self._full_data = json.load(f)
                else:
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        self._full_data = json.load(f)
            except Exception as e:
                print(f"Error loading {self.data_path}: {e}")
                self._full_data = {}
        return self._full_data or {}
    
    def get(self, key, default=None):
        """Cache'li get işlemi"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        full_data = self._load_full_data()
        value = full_data.get(str(key), default)
        
        if value is not None:
            self._add_to_cache(key, value)
        
        return value
    
    def _add_to_cache(self, key, value):
        """Cache'e ekle ve boyutu kontrol et"""
        if len(self.cache) >= self.max_cache_size:
            # En az kullanılan öğeyi kaldır
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            self.cache.pop(least_used, None)
            self.access_count.pop(least_used, None)
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def keys(self):
        """Tüm anahtarları döndür"""
        return self._load_full_data().keys()
    
    def items(self):
        """Tüm öğeleri döndür (generator)"""
        for key, value in self._load_full_data().items():
            yield key, value
    
    def __contains__(self, key):
        """İçerik kontrolü"""
        return str(key) in self._load_full_data()
    
    def clear_cache(self):
        """Cache'i temizle"""
        self.cache.clear()
        self.access_count.clear()
        gc.collect()

def generate_username():
    """Daha az seçenek ile username oluştur"""
    adjectives = ['Cool', 'Swift', 'Bright', 'Smart', 'Epic', 'Black']
    nouns = ['Otaku', 'Ninja', 'Dragon', 'Wolf', 'Fox']
    return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(100, 999)}"

def clean_message(message):
    """Mesaj temizleme - optimize edildi"""
    if not message:
        return ""
    
    # HTML tag'leri temizle
    message = re.sub(r'<[^>]*>', '', message)
    # Uzunluk kontrolü
    if len(message) > 300:  # 500 -> 300
        message = message[:300]
    return message.strip()

def compress_and_save(data, filepath):
    """Veriyi sıkıştırarak kaydet"""
    try:
        with gzip.open(filepath + '.gz', 'wt', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
        return True
    except Exception as e:
        print(f"Error compressing {filepath}: {e}")
        return False

# Sitemap fonksiyonları - optimize edildi
@app.route('/sitemap.xml')
def sitemap():
    """Optimize edilmiş sitemap"""
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
        
        # Sadece popüler anime'leri ekle (50 ile sınırla)
        if recommendation_system and hasattr(recommendation_system, 'id_to_anime'):
            anime_count = 0
            for anime_id in list(recommendation_system.id_to_anime.keys())[:50]:
                if anime_count >= 50:
                    break
                
                try:
                    anime_data = recommendation_system.id_to_anime.get(anime_id)
                    if anime_data:
                        anime_name = anime_data[0] if isinstance(anime_data, list) else str(anime_data)
                        safe_name = re.sub(r'[^\w\-]', '', anime_name.replace(' ', '-'))
                        
                        url = ET.SubElement(urlset, 'url')
                        ET.SubElement(url, 'loc').text = f'{base_url}/anime/{anime_id}/{safe_name}'
                        ET.SubElement(url, 'lastmod').text = current_date
                        ET.SubElement(url, 'changefreq').text = 'weekly'
                        ET.SubElement(url, 'priority').text = '0.6'
                        
                        anime_count += 1
                except Exception:
                    continue
        
        xml_str = ET.tostring(urlset, encoding='unicode')
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        
        return Response(xml_declaration + xml_str, mimetype='application/xml')
        
    except Exception as e:
        print(f"Sitemap error: {e}")
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>',
            mimetype='application/xml'
        )

@app.route('/robots.txt')
def robots_txt():
    """Robots.txt"""
    return Response(f"""User-agent: *
Allow: /
Sitemap: {request.url_root.rstrip('/')}/sitemap.xml
""", mimetype='text/plain')

@app.route('/anime/<int:anime_id>/<path:anime_name>')
def anime_detail(anime_id, anime_name):
    """Anime detay sayfası"""
    if not recommendation_system:
        return render_template('error.html', error="System not available"), 404
    
    anime_data = recommendation_system.id_to_anime.get(str(anime_id))
    if not anime_data:
        return render_template('error.html', error="Anime not found"), 404
    
    anime_name = anime_data[0] if isinstance(anime_data, list) else str(anime_data)
    
    # Basit anime bilgileri
    anime_info = {
        'id': anime_id,
        'name': anime_name,
        'image_url': recommendation_system.get_anime_image_url(anime_id),
        'mal_url': recommendation_system.get_anime_mal_url(anime_id),
        'genres': recommendation_system.get_anime_genres(anime_id),
        'type': recommendation_system._get_type(anime_id)
    }
    
    return render_template('anime_detail.html', anime=anime_info)

@app.route('/chat')
def chat():
    return render_template('chat.html')

# SocketIO events - optimize edildi
@socketio.on('connect')
def on_connect():
    if len(active_users) >= 100:  # Maksimum kullanıcı sınırı
        emit('error', {'message': 'Server is full'})
        return False
    
    username = generate_username()
    active_users[request.sid] = {
        'username': username,
        'connected_at': datetime.now().timestamp()  # datetime yerine timestamp
    }
    
    # Son 20 mesajı gönder
    emit('chat_history', chat_messages[-20:])
    
    join_message = {
        'username': 'System',
        'message': f'{username} joined',
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
            'message': f'{username} left',
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

class AnimeRecommendationSystem:
    """RAM optimize edilmiş öneri sistemi"""
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path, 
                 mal_urls_path, type_seq_path, genres_path):
        self.model = None
        self.dataset = None
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        
        # Memory optimized dictionaries
        self.id_to_anime = MemoryOptimizedDict(animes_path, 500)
        self.id_to_url = MemoryOptimizedDict(images_path, 200)
        self.id_to_mal_url = MemoryOptimizedDict(mal_urls_path, 200)
        self.id_to_genres = MemoryOptimizedDict(genres_path, 300)
        self.id_to_type_seq = MemoryOptimizedDict(type_seq_path, 300)
        
        self.load_model_and_data()
    
    def load_model_and_data(self):
        """Model ve veri yükleme - optimize edildi"""
        try:
            print("Loading model and data...")
            
            # Dataset yükleme
            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)
            
            # Model yükleme
            self.model = model_factory(args)
            self.load_checkpoint()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def load_checkpoint(self):
        """Checkpoint yükleme"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Checkpoint'i bellekten temizle
            del checkpoint
            gc.collect()
            
        except Exception as e:
            raise Exception(f"Failed to load checkpoint: {str(e)}")
    
    def get_anime_genres(self, anime_id):
        """Anime türlerini al"""
        genres = self.id_to_genres.get(anime_id, [])
        return [genre.title() for genre in genres] if genres else []
    
    def get_all_animes(self):
        """Tüm anime listesi - paginated"""
        animes = []
        count = 0
        
        for k in list(self.id_to_anime.keys())[:1000]:  # İlk 1000 anime
            try:
                v = self.id_to_anime.get(k)
                if v:
                    anime_name = v[0] if isinstance(v, list) else str(v)
                    animes.append((int(k), anime_name))
                    count += 1
                    if count >= 1000:
                        break
            except Exception:
                continue
        
        animes.sort(key=lambda x: x[1])
        return animes
    
    def get_anime_image_url(self, anime_id):
        """Anime resim URL'i"""
        return self.id_to_url.get(anime_id)
    
    def get_anime_mal_url(self, anime_id):
        """MAL URL'i"""
        return self.id_to_mal_url.get(anime_id)
    
    def _get_type(self, anime_id):
        """Anime tipi"""
        type_seq_info = self.id_to_type_seq.get(anime_id)
        if type_seq_info and len(type_seq_info) >= 2:
            return type_seq_info[1]
        return "Unknown"
    
    def _is_hentai(self, anime_id):
        """Hentai kontrolü"""
        type_seq_info = self.id_to_type_seq.get(anime_id)
        if type_seq_info and len(type_seq_info) >= 3:
            return type_seq_info[2]
        return False
    
    def get_recommendations(self, favorite_anime_ids, num_recommendations=20, filters=None):
        """Optimize edilmiş öneri sistemi"""
        try:
            if not favorite_anime_ids:
                return [], [], "Please add some favorite animes first!"
            
            # Maksimum öneri sayısını sınırla
            num_recommendations = min(num_recommendations, 40)
            
            smap = self.dataset
            inverted_smap = {v: k for k, v in smap.items()}
            
            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])
            
            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"
            
            # Daha kısa sequence uzunluğu
            target_len = 64  # 128 -> 64
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
            
            # Daha az prediction
            max_predictions = min(200, len(inverted_smap))  # 500 -> 200
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)
            
            recommendations = []
            
            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if len(recommendations) >= num_recommendations:
                    break
                
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]
                    
                    if anime_id in favorite_anime_ids:
                        continue
                    
                    anime_data = self.id_to_anime.get(str(anime_id))
                    if anime_data:
                        # Filtreleme kontrolü
                        if filters and not self._should_include_anime(anime_id, filters):
                            continue
                        
                        anime_name = anime_data[0] if isinstance(anime_data, list) else str(anime_data)
                        
                        recommendations.append({
                            'id': anime_id,
                            'name': anime_name,
                            'score': float(score),
                            'image_url': self.get_anime_image_url(anime_id),
                            'mal_url': self.get_anime_mal_url(anime_id),
                            'genres': self.get_anime_genres(anime_id)
                        })
            
            # Bellekten temizle
            del logits, last_logits, top_scores, top_indices
            gc.collect()
            
            return recommendations, [r['score'] for r in recommendations], f"Found {len(recommendations)} recommendations!"
            
        except Exception as e:
            return [], [], f"Error: {str(e)}"
    
    def _should_include_anime(self, anime_id, filters):
        """Basit filtreleme"""
        if not filters:
            return True
        
        # Blacklist kontrolü
        if 'blacklisted_animes' in filters:
            if anime_id in filters['blacklisted_animes']:
                return False
        
        # Hentai kontrolü
        if 'show_hentai' in filters:
            is_hentai = self._is_hentai(anime_id)
            if not filters['show_hentai'] and is_hentai:
                return False
        
        return True
    
    def clear_caches(self):
        """Tüm cache'leri temizle"""
        self.id_to_anime.clear_cache()
        self.id_to_url.clear_cache()
        self.id_to_mal_url.clear_cache()
        self.id_to_genres.clear_cache()
        self.id_to_type_seq.clear_cache()
        gc.collect()

# Global sistem
recommendation_system = None

@app.route('/')
def index():
    if recommendation_system is None:
        return render_template('error.html', error="System not initialized")
    
    animes = recommendation_system.get_all_animes()
    return render_template('index.html', animes=animes)

@app.route('/api/search_animes')
def search_animes():
    """Optimize edilmiş anime arama"""
    query = request.args.get('q', '').lower()
    animes = []
    count = 0
    
    for k in list(recommendation_system.id_to_anime.keys())[:500]:  # İlk 500'de ara
        if count >= 50:  # Maksimum 50 sonuç
            break
        
        try:
            v = recommendation_system.id_to_anime.get(k)
            if v:
                anime_names = v if isinstance(v, list) else [v]
                main_name = anime_names[0] if anime_names else "Unknown"
                
                if not query or query in main_name.lower():
                    animes.append((int(k), main_name))
                    count += 1
        except Exception:
            continue
    
    animes.sort(key=lambda x: x[1])
    return jsonify(animes)

@app.route('/api/add_favorite', methods=['POST'])
def add_favorite():
    if 'favorites' not in session:
        session['favorites'] = []
    
    # Maksimum favori sınırı
    if len(session['favorites']) >= 20:
        return jsonify({'success': False, 'message': 'Maximum 20 favorites allowed'})
    
    data = request.get_json()
    anime_id = int(data['anime_id'])
    
    if anime_id not in session['favorites']:
        session['favorites'].append(anime_id)
        session.modified = True
        return jsonify({'success': True})
    
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
    
    return jsonify({'success': False})

@app.route('/api/clear_favorites', methods=['POST'])
def clear_favorites():
    session['favorites'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/get_favorites')
def get_favorites():
    if 'favorites' not in session:
        session['favorites'] = []
    
    favorite_animes = []
    for anime_id in session['favorites']:
        anime_data = recommendation_system.id_to_anime.get(str(anime_id))
        if anime_data:
            anime_name = anime_data[0] if isinstance(anime_data, list) else str(anime_data)
            favorite_animes.append({'id': anime_id, 'name': anime_name})
    
    return jsonify(favorite_animes)

@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    if 'favorites' not in session or not session['favorites']:
        return jsonify({'success': False, 'message': 'Please add some favorite animes first!'})
    
    data = request.get_json() or {}
    filters = data.get('filters', {})
    
    # Blacklist bilgisi
    blacklisted_animes = data.get('blacklisted_animes', [])
    if blacklisted_animes:
        filters['blacklisted_animes'] = blacklisted_animes
    
    recommendations, scores, message = recommendation_system.get_recommendations(
        session['favorites'],
        num_recommendations=20,  # Sabit 20 öneri
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

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Cache temizleme endpoint'i"""
    if recommendation_system:
        recommendation_system.clear_caches()
    gc.collect()
    return jsonify({'success': True, 'message': 'Cache cleared'})

def download_from_gdrive(file_id, output_path):
    """Google Drive'dan dosya indir"""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=True)
        print(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def main():
    global recommendation_system
    
    args.num_items = 12689
    
    # Dosya ID'leri
    file_ids = {
        "1C6mdjblhiWGhRgbIk5DP2XCc4ElS9x8p": "pretrained_bert.pth",
        "1U42cFrdLFT8NVNikT9C5SD9aAux7a5U2": "animes.json",
        "1s-8FM1Wi2wOWJ9cstvm-O1_6XculTcTG": "dataset.pkl",
        "1SOm1llcTKfhr-RTHC0dhaZ4AfWPs8wRx": "id_to_url.json",
        "1vwJEMEOIYwvCKCCbbeaP0U_9L3NhvBzg": "anime_to_malurl.json",
        "1_TyzON6ie2CqvzVNvPyc9prMTwLMefdu": "anime_to_typenseq.json",
        "1G9O_ahyuJ5aO0cwoVnIXrlzMqjKrf2aw": "id_to_genres.json"
    }
    
    # Dosyaları indir
    for file_id, filename in file_ids.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            download_from_gdrive(file_id, filename)
    
    try:
        # Sistem başlatma
        recommendation_system = AnimeRecommendationSystem(
            "pretrained_bert.pth",
            "dataset.pkl", 
            "animes.json",
            "id_to_url.json",
            "anime_to_malurl.json",
            "anime_to_typenseq.json",
            "id_to_genres.json"
        )
        
        print("System initialized successfully!")
        
        # Periyodik cache temizleme
        def periodic_cleanup():
            import threading
            import time
            
            while True:
                time.sleep(1800)  # 30 dakikada bir
                if recommendation_system:
                    recommendation_system.clear_caches()
                    gc.collect()
                    print("Periodic cache cleanup completed")
        
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        sys.exit(1)
    
    # Sunucuyu başlat
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
