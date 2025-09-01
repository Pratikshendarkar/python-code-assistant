import streamlit as st
import ast
import sys
import io
import contextlib
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Python Code Assistant",
    page_icon="🐍",
    layout="wide"
)

# Initialize session state
if 'review_history' not in st.session_state:
    st.session_state.review_history = []

class CodeIssue(BaseModel):
    line_number: int = Field(description="Line number where issue occurs")
    issue_type: str = Field(description="Type: bug, style, performance, security, logic")
    severity: str = Field(description="Severity: high, medium, low")
    title: str = Field(description="Brief title of the issue")
    description: str = Field(description="Detailed description of the issue")
    original_code: str = Field(description="The problematic code snippet")
    corrected_code: str = Field(description="The corrected code snippet")
    explanation: str = Field(description="Why this change improves the code")

class CodeReviewResult(BaseModel):
    overall_score: int = Field(description="Overall code quality score (1-10)")
    total_issues: int = Field(description="Total number of issues found")
    issues: List[CodeIssue] = Field(description="List of all issues found")
    suggestions: List[str] = Field(description="General improvement suggestions")
    positive_aspects: List[str] = Field(description="Good aspects of the code")
    summary: str = Field(description="Overall summary of the review")

def setup_llm():
    """Initialize the Language Model"""
    try:
        # Try OpenAI first
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=st.secrets.get("OPENAI_API_KEY", "")
        )
    except:
        st.error("Please add your OpenAI API key to .streamlit/secrets.toml")
        st.stop()

def analyze_syntax(code: str) -> Dict[str, Any]:
    """Basic syntax and structure analysis"""
    issues = []
    try:
        tree = ast.parse(code)
        
        # Count functions, classes, etc.
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        # Check for basic issues
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ['sum', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float']:
                issues.append(f"Line {node.lineno}: Variable '{node.id}' shadows built-in")
        
        return {
            "syntax_valid": True,
            "functions": functions,
            "classes": classes,
            "basic_issues": issues
        }
    except SyntaxError as e:
        return {
            "syntax_valid": False,
            "error": str(e),
            "line": getattr(e, 'lineno', 'Unknown')
        }

def create_review_chain():
    """Create the LangChain for comprehensive code review"""
    
    prompt_template = """
    You are an expert Python code reviewer. Analyze the following Python code thoroughly and provide detailed feedback.

    Python Code:
    ```python
    {code}
    ```

    Analyze the code for:
    1. Bugs and logic errors
    2. PEP 8 style violations
    3. Performance issues
    4. Security vulnerabilities
    5. Best practice violations
    6. Code readability and maintainability
    7. Error handling issues
    8. Type-related problems

    For each issue you find, provide:
    - Exact line number where possible
    - Issue type (bug, style, performance, security, logic)
    - Severity level (high, medium, low)
    - Clear title and description
    - The problematic code snippet
    - A corrected version
    - Explanation of why the change is beneficial

    Also provide:
    - Overall code quality score (1-10)
    - Positive aspects of the code
    - General improvement suggestions
    - Summary of findings

    {format_instructions}
    """
    
    parser = PydanticOutputParser(pydantic_object=CodeReviewResult)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["code"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = setup_llm()
    return LLMChain(llm=llm, prompt=prompt), parser

def create_code_builder_chain():
    """Create chain for helping build/improve code"""
    
    prompt_template = """
    You are an expert Python developer. The user needs help with their Python code.

    User's Request: {request}
    
    Current Code (if provided):
    ```python
    {current_code}
    ```

    Please provide:
    1. Complete, working Python code that addresses their request
    2. Clear explanations of what the code does
    3. Best practices implemented
    4. Error handling where appropriate
    5. Comments explaining complex parts
    6. Suggestions for further improvements

    Make sure the code is:
    - Pythonic and follows PEP 8
    - Robust with proper error handling
    - Well-documented with docstrings
    - Efficient and readable
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["request", "current_code"]
    )
    
    llm = setup_llm()
    return LLMChain(llm=llm, prompt=prompt)

def execute_code_safely(code: str) -> str:
    """Safely execute Python code and return output"""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        # Create a restricted execution environment
        exec_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'max': max,
                'min': min,
                'sum': sum,
                'sorted': sorted,
                'abs': abs,
                'round': round,
            }
        }
        
        exec(code, exec_globals)
        output = captured_output.getvalue()
        return output if output else "Code executed successfully (no output)"
        
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

def main():
    st.title("🐍 Python Code Review & Correction Assistant")
    st.markdown("*Comprehensive Python code analysis, correction, and building assistance*")
    
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Options")
        mode = st.radio(
            "Choose Mode:",
            ["Code Review & Correction", "Code Building Assistant", "Code Execution"]
        )
        
        st.header("📊 Statistics")
        if st.session_state.review_history:
            st.metric("Total Reviews", len(st.session_state.review_history))
            avg_score = sum(r.get('score', 0) for r in st.session_state.review_history) / len(st.session_state.review_history)
            st.metric("Average Score", f"{avg_score:.1f}/10")

    if mode == "Code Review & Correction":
        st.header("📝 Code Review & Correction")
        
        # Code input
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Your Code")
            code_input = st.text_area(
                "Paste your Python code here:",
                height=400,
                placeholder="""def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example usage
result = fibonacci(10)
print(f"The 10th Fibonacci number is: {result}")""",
                key="code_review_input"
            )
        
        with col2:
            st.subheader("Quick Analysis")
            if code_input:
                analysis = analyze_syntax(code_input)
                if analysis["syntax_valid"]:
                    st.success("✅ Syntax is valid")
                    st.info(f"Functions: {analysis['functions']} | Classes: {analysis['classes']}")
                    if analysis["basic_issues"]:
                        st.warning("⚠️ Basic issues detected")
                        for issue in analysis["basic_issues"]:
                            st.text(issue)
                else:
                    st.error(f"❌ Syntax Error: {analysis['error']}")
        
        # Review button
        if st.button("🔍 Review & Correct Code", disabled=not code_input):
            with st.spinner("Analyzing your code..."):
                try:
                    review_chain, parser = create_review_chain()
                    result = review_chain.run(code=code_input)
                    
                    # Simple fallback display if parsing fails
                    st.text_area("Review Result:", result, height=600)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    elif mode == "Code Building Assistant":
        st.header("🛠️ Code Building Assistant")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Your Request")
            user_request = st.text_area(
                "Describe what you want to build or improve:",
                height=200,
                placeholder="I need a function to read CSV files and calculate statistics...",
                key="build_request"
            )
            
            st.subheader("Current Code (Optional)")
            current_code = st.text_area(
                "Paste any existing code:",
                height=200,
                placeholder="# Paste your current code here if you have any",
                key="current_code_input"
            )
        
        with col2:
            st.subheader("Generated Code")
            if st.button("🚀 Build/Improve Code", disabled=not user_request):
                with st.spinner("Building your code..."):
                    try:
                        builder_chain = create_code_builder_chain()
                        result = builder_chain.run(
                            request=user_request,
                            current_code=current_code or "No existing code provided"
                        )
                        
                        st.code(result, language='python')
                        
                    except Exception as e:
                        st.error(f"Error generating code: {str(e)}")

    elif mode == "Code Execution":
        st.header("▶️ Code Execution")
        st.warning("⚠️ This runs code in a restricted environment. Some features may not work.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Code to Execute")
            exec_code = st.text_area(
                "Enter Python code to execute:",
                height=300,
                placeholder="""# Example code
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
print(f"Numbers: {numbers}")
print(f"Average: {average}")""",
                key="exec_code_input"
            )
        
        with col2:
            st.subheader("Output")
            if st.button("▶️ Run Code", disabled=not exec_code):
                output = execute_code_safely(exec_code)
                st.code(output, language='text')

    # Footer
    st.markdown("---")
    st.markdown("**Tips for better results:**")
    st.markdown("• Include complete functions/classes for more accurate analysis")
    st.markdown("• Add comments to help the AI understand your intent")
    st.markdown("• Be specific about what you want to improve or build")

if __name__ == "__main__":
    main()
