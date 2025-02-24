from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from .prepro import Prepro
from .models import Product

class Recommendation:
    def recom(self,df,inp):
        # Text Preprocessing - Convert 'name' to numerical representation
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['name'])

        k = 11  # Number of neighbors, including the item itself
        knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn_model.fit(tfidf_matrix)

        # User input
        user_name = inp

        # Filter products containing the user input in their name
        filtered_products = df[df['name'].str.contains(user_name, case=False)]

        if filtered_products.empty:
            print("No products with the specified keyword found.")
        else:
            # Text Preprocessing for user input
            user_tfidf = tfidf.transform([user_name])

            # Calculate cosine similarity between user input and all products
            similarities, indices = knn_model.kneighbors(user_tfidf)

            # Get top similar products to the user input
            similar_products_indices = indices.flatten()
            similar_products = df.iloc[similar_products_indices]

            # Sort the filtered products by cost and get the top 15
            top_15_products = similar_products.sort_values(by='actual_price').head(15)

            # Get recommended product names, costs, and images from the top 15 products
            recommended_product_names = top_15_products['name'].tolist()
            recommended_product_costs = top_15_products['actual_price'].tolist()
            recommended_product_images = top_15_products['image'].tolist()
        return recommended_product_names,recommended_product_costs,recommended_product_images

    # You can then use these recommended lists as needed


class CollRec:
    def get_product_recommendations(dataset_path='combined_dataset.csv', top_k=10):
        # Load data from the database (assuming it's Django's ORM)
        data = Product.objects.all()

        # Extract necessary fields
        ids = [product.id for product in data]
        id123 = len(ids)

        data1 = Product.objects.filter(id=id123)
        name45 = [item.name for item in data1]
        name45 = name45[0]

        # Assuming 'combined_dataset.csv' contains columns 'name', 'actual_price', 'image'
        df = Prepro.pre(dataset_path)

        # Create a LabelEncoder to encode product names
        label_encoder = LabelEncoder()
        df['name_numeric'] = label_encoder.fit_transform(df['name'])

        # Create a DataFrame with the necessary columns for collaborative filtering
        collab_df = df[['name_numeric', 'actual_price']]

        # Encode the user input into the feature space
        user_input_name_numeric = label_encoder.transform([name45])

        # Create a DataFrame for the user input
        user_input_features = pd.DataFrame([[user_input_name_numeric[0], 0]], columns=['name_numeric', 'actual_price'])

        # Compute cosine similarity between user input and all products
        cosine_sim = cosine_similarity(user_input_features, collab_df)

        # Get indices of the top K similar products
        indices = cosine_sim.argsort()[0][-top_k:][::-1]

        # Get recommended product names, costs, and images based on the nearest neighbors
        recommended_products = df.iloc[indices][['name', 'actual_price', 'image']]

        return recommended_products