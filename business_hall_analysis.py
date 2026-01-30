"""
营业厅场景分析工具
进一步优化：
1. 第一步A：生成人员描述（专门识别工作人员和顾客）
2. 第一步B：生成环境描述（描述营业厅整体环境）
3. 第二步：识别工作人员状态及位置，仅使用人员描述作为参考
4. 第三步：绘制检测框标注工作人员位置
"""
import os
import base64
import json
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor


def encode_image(image_path):
    """将本地图像转换为Base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_json_output(text):
    """解析JSON输出，移除markdown标记"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            text = "\n".join(lines[i + 1:])
            text = text.split("```")[0]
            break
    return text


def analyze_personnel(client, image_base64, image_path, output_txt="personnel_analysis.txt"):
    """
    第一步A：专门分析人员情况
    重点识别工作人员和顾客的身份、位置、行为
    返回: 人员描述文本
    """

    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """逐个分析图中人员，找出工作人员和顾客。
    
**逐个识别图中人员，每次分析人员身份前，请复述【请注意，工作人员位于柜台内（图片下方），穿着工作服；顾客位于柜台外（其他位置），不穿工作服】**

每位人员的详细描述：
     * 位置（柜台内或柜台外，见名词解释）
     * 衣着（是否穿工作服） 
     * 当前行为（使用电脑、使用手机等）

名词解释：
    -柜台内：位于屏幕下方，专属于工作人员，可以看到电脑、办公桌等办公设备，类似于办公室
    -柜台外：位于屏幕中央，顾客等待、接受服务的区域，类似于办事大厅

"""

    completion = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    personnel_description = completion.choices[0].message.content

    # 输出到控制台
    print("\n人员识别与分析结果:")
    print("="*80)
    print(personnel_description)
    print("="*80)

    # 保存到txt文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("人员识别与分析\n")
            f.write("="*80 + "\n\n")
            f.write(personnel_description)
            f.write("\n" + "="*80 + "\n")
        print(f"\n✓ 人员分析已保存到: {output_txt}")
    except Exception as e:
        print(f"\n✗ 保存人员分析文件失败: {e}")

    return personnel_description


def analyze_environment(client, image_base64, image_path, output_txt="environment_analysis.txt"):
    """
    第一步B：专门分析环境情况
    描述营业厅整体环境、设施、氛围等
    返回: 环境描述文本
    """

    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """请详细描述这张营业厅场景图片的环境和氛围，需要包含以下内容：

【环境描述】

1. 营业厅的整体环境
2. 营业厅的设施设备
3. 人员活动情况
"""

    completion = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    environment_description = completion.choices[0].message.content

    # 输出到控制台
    print("\n环境描述结果:")
    print("="*80)
    print(environment_description)
    print("="*80)

    # 保存到txt文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("环境描述\n")
            f.write("="*80 + "\n\n")
            f.write(environment_description)
            f.write("\n" + "="*80 + "\n")
        print(f"\n✓ 环境描述已保存到: {output_txt}")
    except Exception as e:
        print(f"\n✗ 保存环境描述文件失败: {e}")

    return environment_description


def detect_staff_status(client, image_base64, image_path, personnel_description=None):
    """
    第二步：识别工作人员的状态及位置
    仅使用第一步A的人员描述作为参考
    返回相对坐标（归一化到[0, 999]）

    参数:
        personnel_description: 第一步A生成的人员描述
    """
    print("\n" + "="*80)
    print("第二步：识别工作人员状态及位置")
    print("="*80)

    if personnel_description:
        print("\n【参考信息】来自第一步的人员识别分析：")
        print("-"*80)
        print(personnel_description[:800] + "..." if len(personnel_description) > 800 else personnel_description)
        print("-"*80)

    data_url = f"data:image/jpeg;base64,{image_base64}"

    # 构建提示词，仅包含人员识别参考信息
    reference_section = ""
    if personnel_description:
        reference_section = f"""
**【参考信息】**
{personnel_description}
"""

    prompt = f"""结合【参考信息】，识别图片中的所有**工作人员**，并输出坐标：

{reference_section}

**【识别字段】**
1. bbox_2d：bbox_2d格式，坐标值范围0-999
2. label：
   - "使用电脑"：工作人员正在操作电脑
   - "使用手机"：工作人员正在使用手机
   - "正常工作"：工作人员在进行其他工作活动（如接待顾客、整理文件、站立服务等）

**【输出格式】**
请以JSON格式输出：
```json
[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "label": "工作人员-使用电脑"
  }}

]
```
"""

    completion = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    response_text = completion.choices[0].message.content
    print("\n工作人员检测结果:")
    print("="*80)
    print(response_text)
    print("="*80)

    return response_text


def plot_staff_positions(image_path, json_output, output_path="staff_positions_result.jpg"):
    """
    第三步：绘制工作人员位置检测框
    参照draw_position.py绘制相关坐标位置
    """
    print("\n" + "="*80)
    print("第三步：绘制工作人员位置检测框")
    print("="*80)

    # 加载图像
    img = Image.open(image_path)
    width, height = img.size
    print(f"图像尺寸: {img.size}")

    draw = ImageDraw.Draw(img)

    # 定义颜色列表
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
        'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
        'lime', 'navy', 'maroon', 'teal', 'olive', 'coral'
    ]

    # 解析JSON输出
    try:
        json_text = parse_json_output(json_output)
        staff_data = json.loads(json_text)

        if not isinstance(staff_data, list):
            staff_data = [staff_data]

        print(f"\n检测到 {len(staff_data)} 名工作人员")

        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)  # Windows微软雅黑
            except:
                font = ImageFont.load_default()

        # 绘制每个检测框
        for i, staff in enumerate(staff_data):
            color = colors[i % len(colors)]

            # 获取bbox坐标（归一化到0-999）
            bbox = staff.get("bbox_2d", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            # 将归一化坐标(0-999)映射到实际图像尺寸
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)

            # 确保坐标正确
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            print(f"\n工作人员 {i+1}:")
            print(f"  标签: {staff.get('label', 'unknown')}")
            print(f"  坐标(归一化): [{x1}, {y1}, {x2}, {y2}]")
            print(f"  坐标(实际): [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")

            # 绘制矩形框
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)),
                outline=color,
                width=3
            )

            # 添加标签文字
            label = staff.get("label", f"工作人员{i+1}")
            draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)

        # 保存结果图像
        img.save(output_path)
        print(f"\n检测结果已保存到: {output_path}")

        # 显示图像
        img.show()

        return True

    except Exception as e:
        print(f"解析或绘制失败: {e}")
        print(f"原始输出前200字: {json_output[:200]}")
        return False


def main():
    """主函数"""
    # 配置
    image_path = r"E:\python project\qwen-vl\demo5.jpg"
    api_key = "sk-xxx"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    print("="*80)
    print("营业厅场景分析工具")
    print("="*80)

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 编码图像
    print("\n正在读取图像...")
    image_base64 = encode_image(image_path)
    print(f"图像已编码: {len(image_base64)} 字符")

    # 第一步A：生成人员描述
    print("\n" + "█"*80)
    print("第一步A：人员识别与分析")
    print("█"*80)
    personnel_description = analyze_personnel(client, image_base64, image_path)

    # 第一步B：生成环境描述
    print("\n" + "█"*80)
    print("第一步B：环境描述")
    print("█"*80)
    environment_description = analyze_environment(client, image_base64, image_path)

    # 第二步：识别工作人员状态及位置（仅使用人员描述作为参考）
    print("\n" + "█"*80)
    print("第二步：识别工作人员状态及位置（使用人员描述作为参考）")
    print("█"*80)
    staff_detection = detect_staff_status(client, image_base64, image_path,
                                         personnel_description=personnel_description)

    # 第三步：绘制检测框
    print("\n" + "█"*80)
    print("第三步：绘制检测框")
    print("█"*80)
    success = plot_staff_positions(image_path, staff_detection)

    if success:
        print("\n" + "="*80)
        print("✓ 分析完成！")
        print(f"✓ 人员分析已保存到: personnel_analysis.txt")
        print(f"✓ 环境描述已保存到: environment_analysis.txt")
        print(f"✓ 检测结果已保存到: staff_positions_result.jpg")
        print("="*80)
    else:
        print("\n分析过程中出现错误，请检查日志")


if __name__ == "__main__":
    main()
