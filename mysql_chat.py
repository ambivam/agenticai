import streamlit as st
import mysql.connector
from datetime import datetime
import json

# Country mapping dictionary
COUNTRY_MAPPING = {
    # European countries
    'france': 'FRA',
    'england': 'GBR',
    'great britain': 'GBR',
    'uk': 'GBR',
    'united kingdom': 'GBR',
    'britain': 'GBR',
    'germany': 'DEU',
    'italy': 'ITA',
    'spain': 'ESP',
    'netherlands': 'NLD',
    'holland': 'NLD',
    
    # Asian countries
    'india': 'IND',
    'china': 'CHN',
    'japan': 'JPN',
    'south korea': 'KOR',
    'korea': 'KOR',
    'vietnam': 'VNM',
    'thailand': 'THA',
    
    # Americas
    'usa': 'USA',
    'united states': 'USA',
    'america': 'USA',
    'canada': 'CAN',
    'mexico': 'MEX',
    'brazil': 'BRA',
    'argentina': 'ARG',
    
    # Others
    'australia': 'AUS',
    'new zealand': 'NZL',
    'russia': 'RUS',
    'south africa': 'ZAF'
}

# Initialize connection to MySQL
def init_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def save_chat_history(user_input, response):
    conn = init_connection()
    cursor = conn.cursor()
    
    # Create chat_history table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_input TEXT,
            response TEXT,
            timestamp DATETIME
        )
    """)
    
    # Insert the chat exchange
    cursor.execute("""
        INSERT INTO chat_history (user_input, response, timestamp)
        VALUES (%s, %s, %s)
    """, (user_input, response, datetime.now()))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_country_code(country_name):
    """Get the ISO country code for a given country name"""
    return COUNTRY_MAPPING.get(country_name.lower())

def natural_to_sql(query):
    """Convert natural language query to SQL based on common patterns"""
    query = query.lower()
    
    # Extract country name from the query
    for country in COUNTRY_MAPPING.keys():
        if country in query:
            country_code = get_country_code(country)
            if not country_code:
                return None
                
            # Pattern matching for common questions
            if "capital" in query:
                return f"SELECT c.Name FROM city c \
                        INNER JOIN country co ON c.ID = co.Capital \
                        WHERE co.Code='{country_code}'"
            elif "population" in query:
                return f"SELECT Name, Population FROM country \
                        WHERE Code='{country_code}'"
            elif "cities" in query:
                return f"SELECT Name, Population FROM city \
                        WHERE CountryCode='{country_code}' \
                        ORDER BY Population DESC LIMIT 10"
    return None

def format_population(number):
    """Format population with proper suffixes (million/billion)"""
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.2f} billion"
    elif number >= 1_000_000:
        return f"{number/1_000_000:.2f} million"
    else:
        return f"{number:,}"

def get_sql_response(natural_query):
    try:
        # Convert natural language to SQL
        sql_query = natural_to_sql(natural_query)
        if not sql_query:
            return "I don't understand that query. Please try asking about capital cities, population, or major cities of a country."

        conn = init_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Execute the query
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Format the results in a user-friendly way
        if results:
            if len(results) == 1:
                # For single results, return a simple string
                result = results[0]
                if 'Name' in result and 'Population' in result:
                    return f"The population of {result['Name']} is {format_population(result['Population'])}"
                elif 'Name' in result:
                    return f"The answer is: {result['Name']}"
                elif 'Population' in result:
                    return f"The population is {format_population(result['Population'])}"
                else:
                    return json.dumps(result, indent=2, default=str)
            else:
                # For multiple results, format as a list
                response = "Here are the results:\n"
                for result in results:
                    if 'Name' in result and 'Population' in result:
                        response += f"- {result['Name']}: {format_population(result['Population'])}\n"
                    else:
                        response += f"- {json.dumps(result, default=str)}\n"
                return response
        else:
            return "No results found for your query."
            
        cursor.close()
        conn.close()
        
    except mysql.connector.Error as err:
        return f"Error executing query: {err}"

def main():
    st.title("MySQL Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about the database?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        response = get_sql_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        
        # Save to database
        save_chat_history(prompt, response)

if __name__ == "__main__":
    main()
