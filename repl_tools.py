"""Tools for executing code in various languages with safety checks."""

import ast
import sys
import logging
import tempfile
import subprocess
import threading
from typing import Dict, Any, Optional, List
from config import Config

logger = logging.getLogger(__name__)

class CodeExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass

class REPLTools:
    """Tools for executing code in various languages."""
    
    def __init__(self):
        """Initialize REPL tools."""
        self.node_process = None
        self.python_globals = {}
        self.python_locals = {}
    
    def _check_python_safety(self, code: str) -> bool:
        """Check if Python code is safe to execute."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in Config.REPL_BLOCKED_MODULES:
                            raise CodeExecutionError(f"Import of '{name.name}' is not allowed for security reasons")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in Config.REPL_BLOCKED_MODULES:
                        raise CodeExecutionError(f"Import from '{node.module}' is not allowed for security reasons")
                # Check for exec/eval calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['exec', 'eval']:
                        raise CodeExecutionError("Use of exec() or eval() is not allowed")
            return True
        except SyntaxError as e:
            raise CodeExecutionError(f"Python syntax error: {str(e)}")
        except Exception as e:
            raise CodeExecutionError(f"Code safety check failed: {str(e)}")
    
    def execute_python(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute Python code in a safe environment."""
        if not timeout:
            timeout = Config.REPL_TIMEOUT_SECONDS
            
        try:
            # Check code safety
            self._check_python_safety(code)
            
            # Create result container
            result = {
                "output": "",
                "error": None,
                "execution_time": 0
            }
            
            # Create a temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add print redirection
                setup_code = """
import sys
from io import StringIO
output_buffer = StringIO()
sys.stdout = output_buffer
sys.stderr = output_buffer
"""
                f.write(setup_code + "\n" + code)
                
            # Execute in subprocess for isolation
            process = subprocess.Popen(
                [sys.executable, f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                result["output"] = stdout
                if stderr:
                    result["error"] = stderr
                result["execution_time"] = timeout  # Actual time not measured in this version
            except subprocess.TimeoutExpired:
                process.kill()
                raise CodeExecutionError(f"Code execution timed out after {timeout} seconds")
            
            return result
            
        except Exception as e:
            return {
                "output": "",
                "error": str(e),
                "execution_time": 0
            }
    
    def execute_javascript(self, code: str, is_typescript: bool = False, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute JavaScript/TypeScript code using Node.js."""
        if not timeout:
            timeout = Config.REPL_TIMEOUT_SECONDS
            
        try:
            # Determine file extension and Node command
            ext = "ts" if is_typescript else "js"
            cmd = ["ts-node"] if is_typescript else ["node"]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False) as f:
                f.write(code)
            
            # Execute code
            process = subprocess.Popen(
                cmd + [f.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return {
                    "output": stdout,
                    "error": stderr if stderr else None,
                    "execution_time": timeout  # Actual time not measured in this version
                }
            except subprocess.TimeoutExpired:
                process.kill()
                raise CodeExecutionError(f"Code execution timed out after {timeout} seconds")
            
        except Exception as e:
            return {
                "output": "",
                "error": str(e),
                "execution_time": 0
            }
    
    def execute_code(self, code: str, language: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute code in the specified language."""
        if not Config.REPL_ENABLED:
            raise CodeExecutionError("REPL functionality is disabled")
            
        if language not in Config.SUPPORTED_REPL_LANGUAGES:
            raise CodeExecutionError(f"Unsupported language: {language}")
            
        if language == "python":
            return self.execute_python(code, timeout)
        elif language in ["javascript", "typescript"]:
            return self.execute_javascript(code, language == "typescript", timeout)
        else:
            raise CodeExecutionError(f"No executor available for language: {language}")
