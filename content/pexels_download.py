import os
import requests
import re

# === Configuration ===
PEXELS_API_KEY = '65kI3f16hdKwgZCTLgw2DkzWqJ83JiymPAKgZM2VlMSOUVUe8LHR7k0F'  # Replace with your actual API key
SEARCH_QUERY = 'nature'  # Replace with your desired tag
SEARCH_QUERY = 'cat'  # Replace with your desired tag
SEARCH_QUERY = 'dynamic'  # Replace with your desired tag
RESULTS_PER_PAGE = 10
MAX_PAGES = 2
DOWNLOAD_DIR = 'pexels_videos'

# === Setup ===
headers = {
    'Authorization': PEXELS_API_KEY
}
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def sanitize_filename(name):
    """Remove unsafe characters from filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def download_video(video_url, filename):
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {video_url}")

def search_and_download_videos():
    for page in range(1, MAX_PAGES + 1):
        response = requests.get(
            f'https://api.pexels.com/videos/search',
            headers=headers,
            params={
                'query': SEARCH_QUERY,
                'per_page': RESULTS_PER_PAGE,
                'page': page
            }
        )
        if response.status_code != 200:
            print(f"API request failed with status {response.status_code}")
            return

        data = response.json()
        for video in data.get('videos', []):
            # Filter for Full HD resolution (1920x1080)
            full_hd_file = next((f for f in video['video_files'] if f['width'] == 1920 and f['height'] == 1080), None)
            if not full_hd_file:
                continue

            # Create a readable filename from the video's URL
            title_from_url = video['url'].rstrip('/').split('/')[-1]
            safe_name = sanitize_filename(title_from_url)
            filename = os.path.join(DOWNLOAD_DIR, f"{safe_name}.mp4")

            download_video(full_hd_file['link'], filename)

if __name__ == '__main__':
    search_and_download_videos()
