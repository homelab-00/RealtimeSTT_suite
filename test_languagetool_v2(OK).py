# test_languagetool.py

import language_tool_python

def test_languagetool():
    try:
        # Initialize LanguageTool with correct configuration
        tool = language_tool_python.LanguageToolPublicAPI('en-US')
        
        # Sample text with errors
        text = "This are bad sentence."
        
        # Check the text
        matches = tool.check(text)
        
        # Correct the text
        corrected = language_tool_python.utils.correct(text, matches)
        
        print("Original Text:", text)
        print("Corrected Text:", corrected)
        
        # Close the tool
        tool.close()
    except Exception as e:
        print(f"Error initializing LanguageTool: {e}")

if __name__ == '__main__':
    test_languagetool()