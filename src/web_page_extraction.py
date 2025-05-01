import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By


def start_driver():
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    return driver

def get_movie_links(driver):
    """Surinkti visu filmu pavadinimus ir ju nuorodas"""
    movies = driver.find_elements(By.CLASS_NAME, 'ipc-metadata-list-summary-item')

    movie_data = []
    for movie in movies:
        try:
            link_element = movie.find_element(By.CLASS_NAME, 'ipc-lockup-overlay')
            relative_link = link_element.get_attribute('href')
            title_element = movie.find_element(By.CLASS_NAME, 'ipc-title__text')
            title = title_element.text
            movie_data.append({"Title" :  title,
                               "Movie link": relative_link})
        except Exception as e:
            print(f"Elements not found {e}")
    return movie_data

def save_to_json(data, filename='data/IMDb_movie_links.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Data saved to {filename}')

def main():
    driver = start_driver()
    url = 'https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc'
    driver.get(url)
    driver.maximize_window()
    time.sleep(3)

    movies = get_movie_links(driver)
    save_to_json(movies)
    driver.quit()

if __name__ == '__main__':
    main()