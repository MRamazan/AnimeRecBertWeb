from flask import Flask, render_template, request, jsonify, session
import sys
import pickle
from utils import *
from options import args
from models import model_factory
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
import random
import re

app = Flask(__name__)
app.secret_key = '1903bjk'
socketio = SocketIO(app, cors_allowed_origins="*")

chat_messages = []
active_users = {}
MAX_MESSAGES = 300

def generate_username():
    adjectives = ['Cool', 'Awesome', 'Swift', 'Bright', 'Happy', 'Smart', 'Kind', 'Brave', 'Calm', 'Epic']
    nouns = ['Otaku', 'Ninja', 'Samurai', 'Dragon', 'Phoenix', 'Tiger', 'Wolf', 'Eagle', 'Fox', 'Bear']
    return f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(100, 999)}"

def clean_message(message):
    # HTML tag'leri temizle
    message = re.sub(r'<[^>]*>', '', message)
    # Uzunluk kontrolü
    if len(message) > 500:
        message = message[:500]
    return message.strip()


# Chat route'u (mevcut route'lardan sonra ekle)
@app.route('/chat')
def chat():
    return render_template('chat.html')


# SocketIO event'leri (dosyanın sonuna, main() fonksiyonundan önce ekle)
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


class AnimeRecommendationSystem:
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path, mal_urls_path, type_seq_path,
                 genres_path):
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

            args.bert_max_len = 128

            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)

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

            self.model = model_factory(args)
            self.load_checkpoint()

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
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

    def get_filtered_anime_pool(self, filters):
        """Filtrelere göre anime havuzunu önceden filtreler"""
        if not filters:
            return None

        # Eğer sadece hentai istiyorsa, hentai anime'leri filtrele
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
        return type_seq_info[2]  # is_hentai

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

            # Hentai filtresi özel durumu
            filtered_pool = self.get_filtered_anime_pool(filters)
            if filtered_pool is not None:
                return self._get_recommendations_from_pool(favorite_anime_ids, filtered_pool, num_recommendations,
                                                           filters)

            # Normal öneriler
            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            max_predictions = min(500, len(inverted_smap))  # Daha fazla sonuç al

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
                        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(
                            anime_data)
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

            # Anime havuzundaki her anime için skor hesapla
            anime_scores = []
            for anime_id in anime_pool:
                if anime_id in favorite_anime_ids:
                    continue

                if anime_id in smap:
                    model_id = smap[anime_id]
                    if model_id < last_logits.shape[1]:
                        score = last_logits[0, model_id].item()
                        anime_scores.append((anime_id, score))

            # Skorlara göre sırala
            anime_scores.sort(key=lambda x: x[1], reverse=True)

            recommendations = []
            for anime_id, score in anime_scores[:num_recommendations]:
                if str(anime_id) in self.id_to_anime:
                    anime_data = self.id_to_anime[str(anime_id)]
                    anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(
                        anime_data)
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

            return recommendations, [r['score'] for r in
                                     recommendations], f"Found {len(recommendations)} filtered recommendations!"

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

        # Sequel filtresi
        if 'show_sequels' in filters:
            if not filters['show_sequels'] and is_sequel:
                return False

        # Hentai filtresi - özel durum
        if 'show_hentai' in filters:
            if filters['show_hentai']:
                # Sadece hentai göster
                if not is_hentai:
                    return False
            else:
                # Hentai gösterme
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


recommendation_system = None


@app.route('/')
def index():
    if recommendation_system is None:
        return render_template('error.html',
                               error="Recommendation system not initialized. Please check server logs.")

    animes = recommendation_system.get_all_animes()
    return render_template('index.html', animes=animes)


@app.route('/api/search_animes')
def search_animes():
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
    # MyAnimeList logo URL'ini döndür
    return jsonify({
        'success': True,
        'logo_url': 'https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png'
    })


def main():
    global recommendation_system

    args.num_items = 12689

    import gdown
    import os

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
            print(f"Downloading: {file_id}")
            gdown.download(url, output_path, quiet=False)
            print(f"Downloaded: {output_path}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    for key, value in file_ids.items():
        if os.path.isfile(value):
            continue
        download_from_gdrive(key, value)

    try:
        images_path = "id_to_url.json"
        mal_urls_path = "anime_to_malurl.json"
        type_seq_path = "anime_to_typenseq.json"

        if not os.path.exists(images_path):
            print(f"Warning: {images_path} not found. Images will not be displayed.")

        if not os.path.exists(mal_urls_path):
            print(f"Warning: {mal_urls_path} not found. MAL links will not be available.")

        recommendation_system = AnimeRecommendationSystem(
            "pretrained_bert.pth",
            "dataset.pkl",
            "animes.json",
            images_path,
            mal_urls_path,
            type_seq_path,
            "id_to_genres.json"
        )
        print("Recommendation system initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize recommendation system: {e}")
        sys.exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    main()
