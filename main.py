from src import comment_vectorization, comments_extraction, web_page_extraction


def main():
    while True:
        print("\n------IMDb Sentiment Analysis------")
        print("\n1. Scrape top movie links")
        print("2. Scrape user comments")
        print("3. Train and evaluate model")
        print("4. Exit")

        choice = int(input("\nChoose an option (1-4): "))

        if choice == 1:
            print("\nStarting movie link extraction...")
            web_page_extraction.main()
        elif choice == 2:
            print("\nStarting comment extraction...")
            comments_extraction.main()
        elif choice == 3:
            print('\nTraining and evaluating model...')
            comment_vectorization.main()
        elif choice == 4:
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again...")

if __name__ == '__main__':
    main()
