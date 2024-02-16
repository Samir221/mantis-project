import pandas as pd
import numpy as np
import streamlit as st
import openai
from sklearn.metrics.pairwise import cosine_similarity


# Function to display the initial form to enter the OpenAI API key
def get_api_key(container):
    with container.form("api_key_form"):
        api_key = st.text_input('OpenAI API Key', key='api_key_input')
        submitted = st.form_submit_button("Enviar clave API")
        if submitted:
            if api_key:
                st.session_state.api_key = api_key  # Store api_key in session_state
            else:
                st.error("La clave API está vacía. Introduzca una clave API válida.")


# Function to display the form to enter the userID after the API key has been entered
def get_user_id(container):
    with container.form("user_id_form"):
        user_id = st.text_input('User ID', key='user_id_input')
        submitted = st.form_submit_button("Enviar User ID")
        if submitted and user_id:
            st.session_state.user_id = user_id  # Store user_id in session_state
            return user_id
    return None


# Function to filter and prepare data
def prepare_data(userID):
    payments_df = pd.read_csv("../data/payments.csv", sep='|')
    # Replace empty strings with NaN
    payments_df.replace('', np.nan, inplace=True)
    # Then drop rows with all NaN values
    payments_df = payments_df.dropna(how='all')
    docs_df = pd.read_json("../data/userdocuments.json")
    docs_df = docs_df.loc[~((docs_df['content'] == '') | (docs_df['content'] == 'Unsupported file type'))]
    docs_df = docs_df[docs_df['userId'] == userID]
    docs_df = docs_df.drop(["_id", "userId"], axis=1)
    # Remove blank lines from the text in 'TextColumn'
    docs_df['content'] = docs_df['content'].str.replace(r'\n\s*\n', '\n', regex=True)
    return payments_df, docs_df


# Function to vectorize a column in a DataFrame and store the vectors
def vectorize_and_store(df, column_name, file_name):
    # Initialize an empty list to hold the vectors
    vectors = []

    # Loop through each row in the DataFrame and vectorize the text in the specified column
    for text in df[column_name]:
        try:
            # Replace 'text-davinci-002' with the embedding model you wish to use
            response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
            vectors.append(response['data'][0]['embedding'])
        except Exception as e:
            # Handle exceptions (e.g., rate limits, API errors)
            vectors.append(np.zeros(1024))  # Assuming 1024-dimensional embeddings; adjust as needed
            print(f"Error vectorizing text: {e}")

    # Convert the list of vectors into a DataFrame
    vectors_df = pd.DataFrame(vectors)

    # Store the DataFrame as a CSV file
    vectors_df.to_csv(f"{file_name}_vectors.csv", index=False)

    return vectors_df


# Function call after you have the DataFrames 'payments_df' and 'docs_df'
def vectorize_dataframes(payments_df, docs_df):
    # Assuming 'description' is the column to vectorize in payments_df
    payments_vectors_df = vectorize_and_store(payments_df, 'description', 'payments')

    # Assuming 'content' is the column to vectorize in docs_df
    docs_vectors_df = vectorize_and_store(docs_df, 'content', 'documents')


# Function to load vectors from a CSV file
def load_vectors(file_name):
    return pd.read_csv(f"{file_name}.csv")


# Function to embed the user query and return its vector
def embed_query(query):
    try:
        response = openai.Embedding.create(input=query, engine="text-embedding-ada-002")
        return np.array(response['data'][0]['embedding']).reshape(1, -1)  # Reshape for cosine_similarity
    except Exception as e:
        st.error(f"Error embedding query: {e}")
        return None


# Function to calculate similarity and return most similar items
def get_most_similar(vectors_df, query_vector, original_df, top_n=5):
    similarities = cosine_similarity(vectors_df, query_vector).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return original_df.iloc[top_indices]


# Function to generate a natural language response using OpenAI's GPT
def generate_natural_language_response(context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Adjust the model as necessary
            messages=[
                {"role": "system", "content": "Eres un asistente útil. Proporcione únicamente los detalles de la consulta ingresada, no proporcione ninguna información adicional."},
                {"role": "user", "content": context}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"Error al generar respuesta en lenguaje natural: {e}")
        return "Tuve problemas para entender eso. ¿Podrías reformular?"


# Main app logic
def main():
    st.title('Asistente de Finanzas Personales')

    container = st.empty()

    if 'api_key' not in st.session_state:
        get_api_key(container)

    if 'api_key' in st.session_state:
        openai.api_key = st.session_state.api_key

        if 'user_id' not in st.session_state:
            container.empty()
            get_user_id(container)

    if 'user_id' in st.session_state:
        container.empty()
        payments_df, docs_df = prepare_data(st.session_state.user_id)

        vectorize_dataframes(payments_df, docs_df)
        query = st.text_input("Ingresa tu consulta:", key="query_input")
        if query:
            query_vector = embed_query(query)
            if query_vector is not None:
                payments_vectors = load_vectors('payments_vectors')
                docs_vectors = load_vectors('documents_vectors')

                similar_payments = get_most_similar(payments_vectors, query_vector, payments_df)
                similar_docs = get_most_similar(docs_vectors, query_vector, docs_df)

                # Prepare context for GPT from similar entries
                context = "Según su consulta, aquí hay algunos detalles relevantes:\n\n"
                context += "Pagos:\n" + "\n".join([f"- {row['description']}" for _, row in similar_payments.iterrows()]) + "\n\n"
                context += "Documentos:\n" + "\n".join([f"- {row['content']}" for _, row in similar_docs.iterrows()])

                # Generate natural language response
                response = generate_natural_language_response(context)
                st.write(response)

            
if __name__ == '__main__':
    main()
