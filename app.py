import streamlit as st
import pandas as pd
import pickle
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from bing_image_urls import bing_image_urls
# from PIL import Image
# from io import BytesIO
# import requests

st.set_page_config(layout="wide")

# Load model
with open('pickr.pkl', 'rb') as file_1:
    tfidf_df = pickle.load(file_1)

def run():
    
    # Judul
    st.title("Pickr")

    # Subheader
    st.subheader("Beauty Products Recommendation")

    st.write('Products are curated from Sephora Official Website.')

    df = pd.read_csv('sephora_website_clean.csv')

    # Sidebar
    with st.sidebar:

        st.title("Filter your search!")        
        
        # Range slider for price
        st.subheader("PRICE RANGE")
        price_range = st.slider('Price range', 0.0, 550.0, (0.0, 550.0), label_visibility = "collapsed")
        price_filter_down = price_range[0]
        price_filter_up = price_range[1]

        # Checkbox for category
        st.subheader("CATEGORY")
        makeup = st.checkbox('Make-up', True)
        skincare = st.checkbox('Skincare', True)
        bath_body = st.checkbox('Bath & Body', True)
        fragrance = st.checkbox('Fragrance', True)
        hair = st.checkbox('Hair', True)
        tools_brushes = st.checkbox('Tools & Brushes', True)
        mini = st.checkbox('Mini', True)
        gifts = st.checkbox('Gifts', True)
        others = st.checkbox('Others', True)

        category_filter = []

        if makeup:
            category_filter.append('makeup')
        if skincare:
            category_filter.append('skincare')
        if bath_body:
            category_filter.append('bath_body')
        if fragrance:
            category_filter.append('fragrance')
        if hair:
            category_filter.append('hair')
        if tools_brushes:
            category_filter.append('tools_brushes')
        if mini:
            category_filter.append('mini')
        if gifts:
            category_filter.append('gifts')
        if others:
            category_filter.append('others')


        # Multi-select for brand
        st.subheader("BRAND(S)")
        brand_filter = st.multiselect('Brand', df['brand'].unique(), label_visibility = "collapsed")
        if len(brand_filter)==0:
            brand_filter = df['brand'].unique()

        # Radio for rating
        st.subheader("RATING")
        rating_filter = st.radio('Rating', ['0 and above', '1 and above', '2 and above', '3 and above', '4 and above', '5'], label_visibility = "collapsed", index = 0)
        if rating_filter=='0 and above':
            rating_filter = 0
        elif rating_filter=='1 and above':
            rating_filter = 1
        elif rating_filter=='2 and above':
            rating_filter = 2
        elif rating_filter=='3 and above':
            rating_filter = 3
        elif rating_filter == '4 and above':
            rating_filter = 4
        elif rating_filter == '5':
            rating_filter = 5




    # Search page
    with st.container():

        # Search box
        with st.form("User Query"):
            query = st.text_input("What do you want to find?", "Sunscreen")
            # Submit
            submitted = st.form_submit_button('Search')

        if submitted:
            # Recommending based on user query
            df_filter = df[(df['price'] >= price_filter_down) & (df['price'] <= price_filter_up) &
                        (df['rating'] >= rating_filter) & (df['new_category'].isin(category_filter)) & 
                        (df['brand'].isin(brand_filter))] 
            
            if len(df_filter) == 0:
                st.write('Sorry, no product matched your request. Please try other keywords or use the filter on the left of this page.')
            else: 

                # TF-IDF transform df results
                tfidf_df_filter = tfidf_df.transform(df_filter['preprocessing_details_category'])
                
                # Sample query
                blob = TextBlob(query)

                # Corrected query to string
                corrected_query = str(blob.correct())

                # Vectorize query
                corrected_query_vec = tfidf_df.transform([corrected_query])

                # Calculate cosine similarity
                recommendation = cosine_similarity(tfidf_df_filter, corrected_query_vec).reshape((-1, )) 

                # Save result index 
                recommendation_index = []
                for i in recommendation.argsort()[-5:][::-1]:
                    if recommendation[i] >= 0.1:
                        recommendation_index.append(i)
                    else:
                        pass
                if len(recommendation_index) == 0:
                    st.write('Sorry, no product matched your request. Please try other keywords or use the filter on the left of this page.')
                else:
                    # Show data of recommended products
                    df_recommendation = df_filter.iloc[recommendation_index]
                    df_recommendation['url_image'] = 'https://st4.depositphotos.com/14953852/24787/v/450/depositphotos_247872612-stock-illustration-no-image-available-icon-vector.jpg'

                    for i in range(len(df_recommendation)):
                        name = df_recommendation['brand'].iloc[i] + df_recommendation['name'].iloc[i]
                        image_urls = bing_image_urls(name, limit=3)
                        image_url = ''
                        for j in image_urls:
                            if len(j) > 0:
                                image_url = j
                            else:
                                pass
                        df_recommendation['url_image'].iloc[i] = image_url
                    
                    st.write('#### Our recommendations according to your search...')
                    rec_col = st.columns(len(df_recommendation), gap = 'medium')
                    num = 0
                    for col in rec_col:
                        box = col.container(border = True)
                        box.write(f"##### {df_recommendation['brand'].iloc[num]}")
                        if len(df_recommendation['name'].iloc[num]) > 60:
                            box.write(f"{df_recommendation['name'].iloc[num][:57] + '...'}")
                        else:
                            box.write(f"{df_recommendation['name'].iloc[num]}")
                        box.write(f":dollar: ${df_recommendation['price'].iloc[num]} | :star: {df_recommendation['rating'].iloc[num]} | :heart: {df_recommendation['love'].iloc[num]}") #rating
                        box.popover('Product details').write(df_recommendation['details'].iloc[num])
                        box.link_button('Buy here', df_recommendation['url'].iloc[num], type = 'primary')                        
                        try:
                            box.image(df_recommendation['url_image'].iloc[num], use_column_width='auto')
                        except:
                            box.image('https://st4.depositphotos.com/14953852/24787/v/450/depositphotos_247872612-stock-illustration-no-image-available-icon-vector.jpg', use_column_width='always')

                        
                        if num >= len(df_recommendation):
                            break
                        else:
                            num += 1
        else:
            pass

            
            


if __name__ == "__main__":
  run()