import os
from git import Repo


def read_file_diff(file_path):
    try:
        # 转换为绝对路径，如果输入的是绝对路径还是返回绝对路径
        absolute_path = os.path.expanduser(file_path)
        with open(absolute_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"文件路径: {file_path}\n内容片段: {content[:50]}...")
    except UnicodeDecodeError:
        # 处理非文本文件（如图片、二进制文件）
        with open(file_path, 'rb') as f:
            binary_data = f.read()
            print(f"二进制文件: {file_path}, 大小: {len(binary_data)}字节")
    except Exception as e:
        print(f"读取失败: {file_path}, 错误: {e}")


root_path = '/Users/01428674/IdeaProjects/sds-schedule-center/'
repo = Repo(root_path)

commit_a = repo.commit('291d7a0162e95474fafbc06caa6800dde2a34340')  # 前一次提交
commit_b = repo.commit('HEAD')  # 最新提交
diffs = commit_b.diff(commit_a)  # 获取差异对象列表‌

print(commit_a.hexsha)
print(commit_b.hexsha)

for diff in diffs:
    diff_path = diff.a_path
    print(f"变更文件: {diff.a_path}")  # 变更文件路径
    full_path = root_path + diff_path
    read_file_diff(full_path)


