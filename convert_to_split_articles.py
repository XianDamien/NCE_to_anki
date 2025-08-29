import json
import os

def convert_to_split_articles():
    """
    将 cet6_split_data.json 转换为 split_articles 目录下的单独文件
    """
    print("🔄 开始转换数据格式...")
    
    # 定义文件路径
    source_file = os.path.join("processed_data", "cet6_split_data.json")
    output_dir = os.path.join("processed_data", "split_articles")
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"❌ 错误：找不到源文件 {source_file}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 创建输出目录: {output_dir}")
    
    # 读取源数据
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功读取源文件，包含 {len(data)} 个题目")
    except Exception as e:
        print(f"❌ 读取源文件失败: {e}")
        return False
    
    # 转换每个题目
    success_count = 0
    for item in data:
        try:
            # 使用 ID 作为文件名
            output_file = os.path.join(output_dir, f"{item['id']}.json")
            
            # 保存单个题目数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 已创建: {item['id']}.json - {item['title']}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理 {item.get('id', 'unknown')} 失败: {e}")
    
    print(f"\n🏁 转换完成！")
    print(f"   📊 成功处理: {success_count}/{len(data)} 个文件")
    print(f"   📂 输出目录: {output_dir}")
    print(f"   🚀 现在可以运行 2b_generate_notes.py 了！")
    
    return success_count == len(data)

if __name__ == "__main__":
    convert_to_split_articles()
