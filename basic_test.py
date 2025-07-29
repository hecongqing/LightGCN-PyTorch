#!/usr/bin/env python3
"""
基础测试脚本 - 验证代码语法和基本结构
不依赖外部深度学习库
"""

import os
import sys

def test_file_syntax(filename):
    """测试Python文件语法"""
    print(f"测试文件语法: {filename}")
    
    if not os.path.exists(filename):
        print(f"  ❌ 文件不存在: {filename}")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 编译检查语法
        compile(source_code, filename, 'exec')
        print(f"  ✅ 语法检查通过")
        return True
        
    except SyntaxError as e:
        print(f"  ❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  其他错误: {e}")
        return False

def test_imports(filename):
    """测试import语句"""
    print(f"检查导入语句: {filename}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        
        print(f"  发现 {len(import_lines)} 个导入语句")
        for imp in import_lines[:5]:  # 显示前5个
            print(f"    {imp}")
        
        if len(import_lines) > 5:
            print(f"    ... 还有 {len(import_lines) - 5} 个导入语句")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("=== 检查项目结构 ===")
    
    expected_files = [
        'alibaba_dataloader.py',
        'lightgcn_model.py', 
        'train.py',
        'main_alibaba.py',
        'requirements_alibaba.txt',
        'README_LightGCN_Alibaba.md',
        'LightGCN_Alibaba_Tutorial.ipynb'
    ]
    
    existing_files = []
    missing_files = []
    
    for filename in expected_files:
        if os.path.exists(filename):
            existing_files.append(filename)
            print(f"  ✅ {filename}")
        else:
            missing_files.append(filename)
            print(f"  ❌ {filename} (缺失)")
    
    print(f"\n项目文件统计:")
    print(f"  存在: {len(existing_files)}")
    print(f"  缺失: {len(missing_files)}")
    
    return len(existing_files), len(missing_files)

def test_code_structure():
    """测试代码结构"""
    print("\n=== 测试代码结构 ===")
    
    python_files = [
        'alibaba_dataloader.py',
        'lightgcn_model.py', 
        'train.py',
        'main_alibaba.py'
    ]
    
    all_passed = True
    
    for filename in python_files:
        if os.path.exists(filename):
            # 语法测试
            syntax_ok = test_file_syntax(filename)
            
            # 导入测试
            import_ok = test_imports(filename)
            
            if not (syntax_ok and import_ok):
                all_passed = False
            
            print()  # 空行分隔
    
    return all_passed

def analyze_code_metrics():
    """分析代码指标"""
    print("=== 代码指标分析 ===")
    
    python_files = [
        'alibaba_dataloader.py',
        'lightgcn_model.py', 
        'train.py',
        'main_alibaba.py'
    ]
    
    total_lines = 0
    total_comments = 0
    total_classes = 0
    total_functions = 0
    
    for filename in python_files:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        file_lines = len(lines)
        file_comments = sum(1 for line in lines if line.strip().startswith('#') or '"""' in line)
        file_classes = sum(1 for line in lines if line.strip().startswith('class '))
        file_functions = sum(1 for line in lines if line.strip().startswith('def '))
        
        total_lines += file_lines
        total_comments += file_comments
        total_classes += file_classes
        total_functions += file_functions
        
        print(f"{filename}:")
        print(f"  代码行数: {file_lines}")
        print(f"  注释行数: {file_comments}")
        print(f"  类数量: {file_classes}")
        print(f"  函数数量: {file_functions}")
        print()
    
    print("总计:")
    print(f"  总代码行数: {total_lines}")
    print(f"  总注释行数: {total_comments}")
    print(f"  总类数量: {total_classes}")
    print(f"  总函数数量: {total_functions}")
    
    if total_lines > 0:
        comment_ratio = total_comments / total_lines
        print(f"  注释比例: {comment_ratio:.2%}")

def check_documentation():
    """检查文档"""
    print("\n=== 检查文档 ===")
    
    doc_files = [
        'README_LightGCN_Alibaba.md',
        'LightGCN_Alibaba_Tutorial.ipynb'
    ]
    
    for filename in doc_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  ✅ {filename} ({file_size} bytes)")
        else:
            print(f"  ❌ {filename} (缺失)")

def main():
    """主测试函数"""
    print("🧪 LightGCN阿里推荐系统基础测试")
    print("=" * 60)
    
    # 检查项目结构
    existing_count, missing_count = check_project_structure()
    
    # 测试代码结构
    code_structure_ok = test_code_structure()
    
    # 分析代码指标
    analyze_code_metrics()
    
    # 检查文档
    check_documentation()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"  项目文件完整度: {existing_count}/{existing_count + missing_count}")
    print(f"  代码结构: {'✅ 通过' if code_structure_ok else '❌ 有问题'}")
    
    if existing_count >= 4 and code_structure_ok:
        print("🎉 基础测试通过！代码结构良好。")
        return True
    else:
        print("⚠️  存在一些问题，请检查上述输出。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)