import json

from llama_index.core.node_parser import CodeSplitter
import os

file_suffix_dict = {
    '.sh': 'bash',
    '.c': 'c',
    '.cs': 'c_sharp',
    '.cpp': 'cpp',
    '.css': 'css',
    '.dockerfile': 'dockerfile',
    '.dot': 'dot',
    '.el': 'elisp',
    '.ex': 'elixir',
    '.elm': 'elm',
    '.erl': 'erlang',
    '.f90': 'fortran',
    '.go': 'go',
    '.mod': 'gomod',
    '.hh': 'hack',
    '.hs': 'haskell',
    '.hcl': 'hcl',
    '.html': 'html',
    '.java': 'java',
    '.js': 'javascript',
    '.jsdoc': 'jsdoc',
    '.json': 'json',
    '.jl': 'julia',
    '.kt': 'kotlin',
    '.lua': 'lua',
    '.mk': 'make',
    '.md': 'markdown',
    '.m': 'objc',
    '.ml': 'ocaml',
    '.pl': 'perl',
    '.php': 'php',
    '.py': 'python',
    '.ql': 'ql',
    '.r': 'r',
    '.regex': 'regex',
    '.rst': 'rst',
    '.rb': 'ruby',
    '.rs': 'rust',
    '.scala': 'scala',
    '.sql': 'sql',
    '.sqlite': 'sqlite',
    '.toml': 'toml',
    '.tsq': 'tsq',
    '.ts': 'typescript',
    '.yaml': 'yaml'
}

care_extensions = {
    ".java"
}

def read_all_files(root_dir):
    file_num = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            this_root, ext = os.path.splitext(file_path)
            lang = file_suffix_dict.get(ext)
            if ext not in care_extensions:
                print(f"文件: {file_path}无需关注")
                continue
            if lang is None:
                print(f"文件: {file_path}无需关注")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    split_content(content, lang)
                    print(f"文件路径: {file_path}\n内容片段: {content[:50]}...")
            except UnicodeDecodeError:
                # 处理非文本文件（如图片、二进制文件）
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                    print(f"二进制文件: {file_path}, 大小: {len(binary_data)}字节")
            except Exception as e:
                print(f"读取失败: {file_path}, 错误: {e}")

            file_num = file_num + 1

    return file_num

def split_content(content, lang):
    split_contents = CodeSplitter.from_defaults(lang, 20, 6, 1200).split_text(content)
    print(len(split_contents))
    return split_contents


if __name__ == '__main__':
    file_num = read_all_files("/Users/01428674/Downloads/test")
    print(file_num)
