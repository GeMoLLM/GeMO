import json
import os
import random
from tqdm import tqdm

titles = os.listdir('../review_data/goodreads/grouped_reviews_long_sub_en')
print(len(titles))

def sample_long_reviews_count_10(source_folder='../review_data/goodreads/grouped_reviews', target_folder='../review_data/goodreads/grouped_reviews_long_sub_en_10', min_word_length=300, max_word_length=700, sample_size=10):
    os.makedirs(target_folder, exist_ok=True)  # Ensure the target directory exists

    for file_name in tqdm(titles):
        file_path = os.path.join(source_folder, file_name)
        valid_reviews = []  # List to hold reviews that meet the word length criterion and are not in Spanish

        # Read and filter the reviews from the source file
        with open(file_path, 'r') as file:
            for line in file:
                review = json.loads(line)
                review_text = review['review_text']
                word_count = len(review_text.split())

                if max_word_length >= word_count >= min_word_length:
                    valid_reviews.append(review)

        # Proceed only if there are more than 5 reviews meeting the criteria
        if len(valid_reviews) >= sample_size:
            sampled_reviews = random.sample(valid_reviews, sample_size)  # Randomly sample 5 reviews

            # Write the sampled reviews to a new file in the target folder
            target_file_path = os.path.join(target_folder, file_name)
            with open(target_file_path, 'w') as target_file:
                for review in sampled_reviews:
                    json.dump(review, target_file)
                    target_file.write('\n')

# Call the function to process the reviews
sample_long_reviews_count_10()
