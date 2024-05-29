import pycode_similar
referenced_code_str = "def foo():\n    print('Hello, world!')"
candidate_code_str1 = "def foo():\n    print('Hello, world!')"
candidate_code_str2 = "def bar():\n    print('Hello, world!')"
print(pycode_similar.detect([referenced_code_str, candidate_code_str1, candidate_code_str2], diff_method=pycode_similar.UnifiedDiff, keep_prints=False, module_level=False))
