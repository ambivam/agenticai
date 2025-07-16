"""REPL interface for code execution."""

import streamlit as st
from typing import Optional
import logging
from repl_tools import REPLTools, CodeExecutionError
from config import Config

logger = logging.getLogger(__name__)

def handle_repl_interface():
    """Handle the REPL interface section of the application."""
    try:
        st.header("💻 Code Playground")
        
        # Initialize REPL tools
        if "repl_tools" not in st.session_state:
            st.session_state.repl_tools = REPLTools()
        
        # Language selector
        language = st.selectbox(
            "Select Language:",
            Config.SUPPORTED_REPL_LANGUAGES,
            key="repl_language"
        )
        
        # Code editor
        if "code_input" not in st.session_state:
            if language == "python":
                default_code = '''# Example Python code
import numpy as np
import pandas as pd

# Create sample data
data = {
    'A': np.random.rand(5),
    'B': np.random.rand(5)
}
df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
'''
            else:
                default_code = '''// Example JavaScript/TypeScript code
function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log("First 5 Fibonacci numbers:");
for (let i = 0; i < 5; i++) {
    console.log(fibonacci(i));
}
'''
            st.session_state.code_input = default_code
        
        # Code editor with syntax highlighting
        st.session_state.code_input = st.text_area(
            "Code Editor:",
            value=st.session_state.code_input,
            height=300,
            key="code_editor"
        )
        
        # Execution options
        col1, col2 = st.columns([3, 1])
        with col1:
            timeout = st.slider(
                "Execution Timeout (seconds):",
                min_value=1,
                max_value=Config.REPL_TIMEOUT_SECONDS,
                value=10
            )
        
        # Execute button
        if st.button("▶️ Run Code", type="primary", use_container_width=True):
            if st.session_state.code_input.strip():
                with st.spinner("Executing code..."):
                    try:
                        result = st.session_state.repl_tools.execute_code(
                            st.session_state.code_input,
                            language,
                            timeout
                        )
                        
                        # Display results
                        if result["error"]:
                            st.error(f"Error:\n```\n{result['error']}\n```")
                        
                        if result["output"]:
                            st.code(result["output"], language=language)
                            
                        # Show execution time
                        if result["execution_time"]:
                            st.info(f"Execution time: {result['execution_time']:.2f} seconds")
                            
                    except CodeExecutionError as e:
                        st.error(f"Execution error: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter some code to execute")
        
        # Help section
        with st.expander("ℹ️ REPL Help"):
            st.markdown("""
            ### Code Playground Help
            
            This is a secure environment for executing code in Python and JavaScript/TypeScript.
            
            #### Features:
            - Syntax highlighting
            - Execution timeout protection
            - Security restrictions on dangerous operations
            
            #### Available Packages:
            **Python:**
            - numpy, pandas, matplotlib, seaborn
            - sklearn, scipy, math, random, datetime
            
            **JavaScript/TypeScript:**
            - lodash, moment, axios
            - react, vue (for framework examples)
            
            #### Security Notes:
            - System operations (file/network access) are restricted
            - Code execution is isolated in a sandbox
            - Maximum execution time is configurable
            """)
    
    except Exception as e:
        logger.error(f"Error in REPL interface: {str(e)}")
        st.error(f"An error occurred in the REPL interface: {str(e)}")
