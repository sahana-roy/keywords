from flask import Flask, render_template, request, jsonify
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

app = Flask(__name__)

# Read and preprocess data (similar to the previous examples)
file_path = 'AssociationAnalysis.xlsx'  # Update with your file path
df = pd.read_excel(file_path)
df['Sureshot Keywords'] = df['Sureshot Keywords'].astype(str)
transactions = df['Sureshot Keywords'].apply(lambda x: [keyword.strip() for keyword in x.split(',')])

# Calculate Item Frequencies
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for processing suggestions
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    user_input_keywords = request.form['user_input_keywords'].lower().split(',')

    co_occurring_keywords = []
    co_occurring_companies = set()

    for idx, transaction in enumerate(transactions):
        if all(keyword in map(str.lower, transaction) for keyword in user_input_keywords):
            co_occurring_keywords.extend(keyword for keyword in transaction if keyword not in user_input_keywords)
            co_occurring_companies.add(df['Sureshot Company'].iloc[idx])

    co_occurring_item_frequencies = pd.Series(co_occurring_keywords).value_counts().index.tolist()

    # Exclude blank suggestions
    top_3_suggestions = [suggestion for suggestion in co_occurring_item_frequencies[:3] if suggestion.strip()]

    # Return suggestions and companies as JSON
    return jsonify({
        'top_suggestions': top_3_suggestions,
        'companies': list(co_occurring_companies)
    })

if __name__ == '__main__':
    app.run(debug=False)
