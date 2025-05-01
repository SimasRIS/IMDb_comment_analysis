import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By


def start_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver


def load_film_links(file_path="data/IMDb_movie_links.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load file {file_path}: {e}")
        return None


def scrape_reviews(driver, movie_url):
    """Return a list of dicts with Rating, Title and Comment for one movie."""
    reviews_data = []
    time.sleep(2)
    driver.get(movie_url)
    time.sleep(2)

    try:
        reviews_link = driver.find_element(By.PARTIAL_LINK_TEXT, 'User reviews')
        driver.get(reviews_link.get_attribute('href'))
        time.sleep(3)

        # Hide spoilers if possible
        try:
            driver.find_element(By.ID, 'title-reviews-hide-spoilers').click()
            time.sleep(2)
        except Exception:
            pass  # Button may not exist

        last_review_count = 0
        time.sleep(0.5)
        while True:
            # Scroll to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

            # Try clicking "Load More" button if available
            try:
                load_more = driver.find_element(By.CLASS_NAME, 'ipc-see-more__button')
                if load_more.is_displayed():
                    driver.execute_script('arguments[0].click();', load_more)
                    time.sleep(2)
            except Exception:
                pass  # No button found

            # Grab reviews
            reviews_block = driver.find_elements(By.CLASS_NAME, 'ipc-list-card__content')
            time.sleep(0.2)
            for review in reviews_block:
                try:
                    rating_el = review.find_elements(By.CLASS_NAME, "ipc-rating-star--rating")
                    rating = rating_el[0].text if rating_el else None
                    title = review.find_element(By.CLASS_NAME, 'ipc-title__text').text
                    review_text = review.find_element(
                        By.CLASS_NAME, "ipc-html-content-inner-div"
                    ).text.strip()

                    if review_text not in [r["Comment"] for r in reviews_data]:
                        reviews_data.append({"Rating": rating, "Title": title, "Comment": review_text})
                except Exception:
                    continue  # Skip bad blocks
            time.sleep(0.2)
            # Stop if reached 2000 comments
            if len(reviews_data) >= 2000:
                print(f"Reached {len(reviews_data)} reviews for {movie_url}.")
                break
            time.sleep(0.2)
            # Stop if no new reviews loaded
            if len(reviews_data) == last_review_count:
                break
            else:
                last_review_count = len(reviews_data)

    except Exception as e:
        print(f"Could not scrape reviews from {movie_url}: {e}")
    time.sleep(1.5)
    return reviews_data


def main():
    driver = start_driver()
    movies = load_film_links()
    all_reviews = []
    driver.maximize_window()

    if not movies:
        print("No movies found. Exiting.")
        driver.quit()
        return

    for movie in movies:
        print(f"Processing movie {movie['Title']}")
        movie_reviews = scrape_reviews(driver, movie['Movie link'])
        all_reviews.append({
            'title': movie['Title'],
            'link': movie['Movie link'],
            'reviews': movie_reviews
        })
        time.sleep(1)
    time.sleep(0.5)
    with open('../data/IMDbs_reviews_2000.json', 'w', encoding='utf-8') as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=4)
        print(f'Data saved to file "IMDbs_reviews.json"')

    driver.quit()


if __name__ == "__main__":
    main()