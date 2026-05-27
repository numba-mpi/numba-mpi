""" extracts code from README.md into a file (name expected as argument) """
import sys
import pytest_codeblocks

assert len(sys.argv) == 2

code = pytest_codeblocks.extract_from_file('README.md')
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.writelines(block.code for block in code if block.syntax=='python')
