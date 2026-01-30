"""
员工行为检测工具
检测四种违规行为：玩手机、睡觉、不穿工服、岗位无人
"""
import os
import base64
import json
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ==================== 全局配置 ====================
# 使用的AI模型名称
AI_MODEL = "qwen3-vl-235b-a22b-instruct"# qwen3-vl-32b-instruct qwen3-vl-plus  qwen3-vl-235b-a22b-instruct
AI_MODEL_CHAT = "qwen3-32b"
# =================================================


def encode_image(image_path):
    """将本地图像转换为Base64编码，返回(base64_data, mime_type)"""
    # 先验证图片是否可以打开
    try:
        with Image.open(image_path) as img:
            # 验证图片尺寸
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"图片尺寸无效: {img.size}")

            # 获取图片格式对应的MIME类型
            format_map = {
                'JPEG': 'image/jpeg',
                'JPG': 'image/jpeg',
                'PNG': 'image/png',
                'BMP': 'image/bmp',
                'GIF': 'image/gif'
            }
            mime_type = format_map.get(img.format.upper(), 'image/jpeg')
    except Exception as e:
        raise ValueError(f"图片验证失败 {image_path}: {e}")

    # 读取并编码
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")

    return base64_data, mime_type


def parse_json_output(text):
    """解析JSON输出，移除markdown标记"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            text = "\n".join(lines[i + 1:])
            text = text.split("```")[0]
            break
    return text


def analyze_workstation(client, image_base64, image_path, output_txt="workstation_analysis.txt"):
    """
    第一步：分析工位状态
    描述工位的情况，不涉及人员分析
    返回: 工位描述文本
    """
    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """你正在从监控视角看营业厅，柜台把画面分成两部分，柜台内近处是工作人员区域，柜台外远处是顾客区域，两个区域被柜台隔开。

先依照【柜台解释】识别柜台位置，然后识别**柜台内工位**总数，最后按照【描述要求】，逐一简要工位情况。

开始回答前，请复述“我应该牢记【柜台解释】中的内容，准确区分柜台内和柜台外，精准识别柜台内的工位是否有工作人员”

**请注意，仅分析柜台内的工位，不要分析柜台外的顾客座位**

柜台位置识别：依照【柜台解释】，识别出柜台的位置，简要输出即可。

工位总数：参照【柜台描述】，明确指出柜台内的工位总数，必须位于“柜台内”

【描述要求】：
     每个工位隔一行输出
     针对每个工位，先描述工位情况，例如整体位置、办公设备、内部人员情况等，
     - 1 工位的位置：是否位于柜台内、位于图像的哪个位置（如左下方）
     - 2 是否属于工作人员：被电脑，打印机等设备包围，离摄像头更近，有办公属性，并不是柜台外的顾客等待区域
     - 3 工位内人员情况：工作人员坐在座椅上、站立等


【柜台解释】：
    - 柜台：物理隔断，把整个大厅分成工作人员区域与顾客区域，工作人员在柜台内侧的办公桌上工作，顾客在柜台外侧
    - 柜台内：在柜台的内侧，靠近摄像头，专属于工作人员，可以看到电脑、办公桌等办公设备，类似于办公室
    - 柜台外：在柜台的外侧，远离摄像头区域，顾客办理业务、等待的区域 
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
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

    workstation_description = completion.choices[0].message.content

    # 输出到控制台
    print("\n工位状态分析结果:")
    print("="*80)
    print(workstation_description)
    print("="*80)

    # 保存到txt文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("工位状态分析\n")
            f.write("="*80 + "\n\n")
            f.write(workstation_description)
            f.write("\n" + "="*80 + "\n")
        print(f"\n✓ 工位状态分析已保存到: {output_txt}")
    except Exception as e:
        print(f"\n✗ 保存工位状态分析文件失败: {e}")

    return workstation_description


def analyze_personnel_behavior(client, image_base64, image_path, output_txt="behavior_analysis.txt"):
    """
    第二步：分析人员行为
    描述工作人员的行为状态，不涉及坐标
    返回: 人员行为描述文本
    """
    data_url = f"data:image/jpeg;base64,{image_base64}"

    prompt = """你正在从监控视角看营业厅，柜台把画面分成两部分，柜台内近处是工作人员区域，柜台外远处是顾客区域，两个区域被柜台隔开。

先识别图中共有几人，再依照【柜台解释】识别柜台位置，然后严格按照【描述要求】，参考【衣着解释】、【身份解释】、【手机解释】，简要分析图中人员行为，找出工作人员和顾客。

开始回答前，请复述“我应该牢记【柜台解释】、【衣着解释】、【身份解释】、【手机解释】中的内容，准确区分柜台内、柜台外，准确区分顾客和工作人员”

**请注意，工作人员位于柜台内（图片下方）；顾客位于柜台外（其他位置）**

总人数识别：识别出图片中有几个人。

柜台位置识别：依照【柜台解释】，识别出柜台的位置，简要输出即可。

【描述要求】：
     针对每个人员，先描述人的整体行为，然后接下从7个方面来描述、判断：
     - 1 身份：顾客或工作人员，（屏幕下方、柜台内是工作人员），参见【身份解释】
     - 2 柜台对面人员：如果是工作人员，请结合柜台位置识别结果，看看柜台正对面的顾客情况
     - 3 位置：例如位于柜台内、画面的左下角等，参见【柜台解释】
     - 4 衣着：介绍人员衣着情况，便服或正式服装参见【衣着解释】
     - 5 手机：直接使用手机，且**柜台对面没有顾客**，**不包括座机/固定电话**，参见【手机解释】；
     - 6 聊天：聊天、交头接耳的行为，**仅限工作人员之间聊天，周围顾客不算**，
     - 7 睡觉：整个上身趴在桌子上，呈休息姿态，例如**头直接趴在桌子上**，或“头枕着胳膊，胳膊放在桌面上”

【柜台解释】：
    - 柜台：物理隔断，把整个大厅分成工作人员区域与顾客区域，工作人员在柜台内侧的办公桌上工作，顾客在柜台外侧
    - 柜台内：在柜台的内侧，靠近摄像头，专属于工作人员，可以看到电脑、办公桌等办公设备，类似于办公室
    - 柜台外：在柜台的外侧，远离摄像头区域，顾客办理业务、等待的区域

【衣着解释】：
    - 正式服装：黑、白、粉等纯色衬衫、西服等，允许褶皱，不用考虑工牌，较为正式、简约
    - 便服：有图案装饰的衣服，偏向生活，较为随意，

【身份解释】：
   - 顾客：一般坐在柜台外（参见【柜台解释】），对面是工作人员，接受服务，不会被办公设备包围；
   - 工作人员：一般坐在柜台内（参见【柜台解释】），为顾客提供服务，一般比顾客更靠近摄像头，被办公设备包围；

【手机解释】：
   - 请准确区分手机和座机，座机有绳子，话筒和桌面上的座机连在一起，属于固定电话
   - 务必看到手机，注意不要把固定电话的话筒识别成手机
   - 周围有顾客则不算是用手机，因为可能正在给用户查资料
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
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

    behavior_description = completion.choices[0].message.content

    # 输出到控制台
    print("\n人员行为分析结果:")
    print("="*80)
    print(behavior_description)
    print("="*80)

    # 保存到txt文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("人员行为分析\n")
            f.write("="*80 + "\n\n")
            f.write(behavior_description)
            f.write("\n" + "="*80 + "\n")
        print(f"\n✓ 行为分析已保存到: {output_txt}")
    except Exception as e:
        print(f"\n✗ 保存行为分析文件失败: {e}")

    return behavior_description


def detect_workstation_violations(client, image_base64, image_path, workstation_description, mime_type="image/jpeg"):
    """
    第三步：根据工位状态描述，检测工位违规行为
    返回: JSON格式的检测结果

    参数:
        workstation_description: 第一步生成的工位状态描述
    """
    print("\n" + "="*80)
    print("第三步：检测工位违规行为并标注坐标")
    print("="*80)

    if workstation_description:
        print("\n【参考信息】来自第一步的工位状态分析：")
        print("-"*80)
        print(workstation_description[:400] + "..." if len(workstation_description) > 400 else workstation_description)
        print("-"*80)

    data_url = f"data:{mime_type};base64,{image_base64}"

    # 构建提示词，仅针对工位违规
    reference_section = ""
    if workstation_description:
        reference_section = f"""
**【参考信息】**
{workstation_description}
"""

    prompt = f"""仔细阅读【参考信息】中每一个的描述，按照【柜台解释】，现招代属于工作人员的工位，然后检测柜台内的**岗位无人**情况：

{reference_section}

**注意**：如果工作人员站在工位附近，可能是工作人员起身交流，不能算“岗位无人”

开始回答前，请复述“我应该牢记【柜台解释】、【识别规则】中的内容，精准识别柜台内的无人工位”

【检测类型】：
     - **岗位无人**：工作人员不在工位内，例如空的座位、空着的办公桌，工位内没有站立的工作人员；

**【识别规则】**
    - 只检测**柜台内**的无人工位（参照【柜台解释】）
    - 该位置必须属于工作人员，被办公设备包围，非顾客等待区域
    - 如果工作人员在工位旁边站着，则不能算工位无人
    - 坐标范围：0-999

【柜台解释】：
    - 柜台：物理隔断，把整个大厅分成工作人员区域与顾客区域，工作人员在柜台内侧的办公桌上工作，顾客在柜台外侧
    - 柜台内：在柜台的内侧，靠近摄像头，专属于工作人员，可以看到电脑、办公桌等办公设备，类似于办公室
    - 柜台外：在柜台的外侧，远离摄像头区域，顾客办理业务、等待的区域

**【输出格式】**
请以JSON格式输出检测结果：
```json
[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "violation_type": "岗位无人"
  }}
]
```

**说明**：
- 对于空工位，需要将空的座位或办公桌区域框选出来
- 如果没有检测到无人工位，请返回空数组[]
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
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
    print("\n工位违规检测结果:")
    print("="*80)
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    print("="*80)

    return response_text


def simplify_staff_description(client, behavior_description, output_txt="staff_simplified.txt"):
    """
    第三步半：简化人员描述
    使用大语言模型处理第二步的人员行为描述，剔除顾客，只保留工作人员信息
    返回: 简化后的人员描述文本
    """
    print("\n" + "="*80)
    print("第三步半：简化人员描述（提取工作人员信息）")
    print("="*80)

    if behavior_description:
        print("\n【原始信息】来自第二步的人员行为分析：")
        print("-"*80)
        print(behavior_description[:400] + "..." if len(behavior_description) > 400 else behavior_description)
        print("-"*80)

    prompt = f"""你正在提取人员描述中的异常行为。请阅读以下【人员行为】，依据【任务要求】和【违规类型】，并完成简化和筛选任务。
    
 在回复过程中，请先复述"我应该仔细阅读【人员行为】中每一个人，分别检测是否有玩手机、聊天、睡觉、便服的行为，不能出现遗漏"


【人员行为】：
{behavior_description}

【任务要求】：
    对每个人员，只保留以下关键信息：
   - 位置（在柜台内的具体位置）
   - 身份（顾客或工作人员）
   - 当前行为（简要描述行为状态）
   - 潜在违规行为（根据以下【违规类型】判断）

【违规类型】：
    - **玩手机**：仅针对工作人员，且柜台正对面无顾客时，正在使用手机，需要能看见手机（包括拿在手中看、操作、打电话等），不包括固定电话
    - **聊天**：仅限工作人员之间聊天，且柜台正面无顾客时
    - **睡觉**：仅针对工作人员，明显的睡眠状态，头趴在桌子上，或头趴在胳膊且胳膊放在桌子上
    - **便服**：仅针对工作人员，穿便服，不穿统一的正式服装（正式服装：黑、白、粉等纯色衬衫、西服等）

【输出格式要求】：
    - 按人员逐个输出，每个人员一段，隔行输出

【示例格式】：
    人员x：位于柜台xxxx，身份是xxx，正在xxxx，未发现违规行为/发现xxx行为
"""

    # 使用流式处理
    completion = client.chat.completions.create(
        model=AI_MODEL_CHAT,  # 使用chat模型
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        extra_body={"enable_thinking": False},  # 明确禁用思考模式
        stream=True,
    )

    # 处理流式响应
    content_parts = []
    for chunk in completion:
        if chunk.choices:
            content = chunk.choices[0].delta.content or ""
            content_parts.append(content)

    simplified_description = "".join(content_parts)

    # 输出到控制台
    print("\n简化后的人员描述（仅工作人员）:")
    print("="*80)
    print(simplified_description)
    print("="*80)

    # 保存到txt文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("简化后人员描述（仅工作人员）\n")
            f.write("="*80 + "\n\n")
            f.write(simplified_description)
            f.write("\n" + "="*80 + "\n")
        print(f"\n✓ 简化后人员描述已保存到: {output_txt}")
    except Exception as e:
        print(f"\n✗ 保存简化后人员描述文件失败: {e}")

    return simplified_description


def detect_personnel_violations(client, image_base64, image_path, simplified_behavior_description, mime_type="image/jpeg"):
    """
    第四步：根据简化后的人员行为描述，检测人员违规行为
    返回: JSON格式的检测结果

    参数:
        simplified_behavior_description: 第三步半生成的简化后人员行为描述（仅含工作人员）
    """
    print("\n" + "="*80)
    print("第四步：检测人员违规行为并标注坐标")
    print("="*80)

    if simplified_behavior_description:
        print("\n【参考信息】来自第三步半的简化后人员描述：")
        print("-"*80)
        print(simplified_behavior_description[:400] + "..." if len(simplified_behavior_description) > 400 else simplified_behavior_description)
        print("-"*80)

    data_url = f"data:{mime_type};base64,{image_base64}"

    # 构建提示词，仅针对人员违规
    reference_section = ""
    if simplified_behavior_description:
        reference_section = f"""
**【参考信息】**
{simplified_behavior_description}
"""

    prompt = f"""，阅读【参考信息】中每一个工作人员的描述，检测图片中的**工作人员**的违规情况。检测前，请确认对应位置是否真的有人。

在回复过程中，请先复述"我应该仔细阅读【参考信息】中每一个人员的描述，严格按照【检测类型】、【柜台解释】、【手机使用规则】确定人员违规行为，不能误识别；"

{reference_section}

**【检测类型】**
    - **玩手机**：仅针对工作人员，且柜台对面没有顾客，参见【手机使用规则】
    - **聊天**：仅限工作人员之间聊天，且柜台对面没有顾客
    - **睡觉**：仅针对工作人员，明显的睡眠状态，上身趴在桌子上
    - **便服**：仅针对工作人员，穿便服，不穿统一的正式服装

【柜台解释】：
    - 柜台：物理隔断，把整个大厅分成工作人员区域与顾客区域，工作人员在柜台内侧的办公桌上工作，顾客在柜台外侧
    - 柜台内：在柜台的内侧，靠近摄像头，专属于工作人员，可以看到电脑、办公桌等办公设备，类似于办公室
    - 柜台外：在柜台的外侧，远离摄像头区域，顾客办理业务、等待的区域
    
【手机使用规则】：
    - 工作人员正对面的顾客区域有人时，不要进行检测
    - 需要能看见手机（包括拿在手中、放在桌子上、打电话等）
    - 不包括固定电话，固定电话一般有绳子。
    
**【识别规则】**
   - 只检测**柜台内**的工作人员（参见【柜台解释】），不要检测顾客
   - 坐标范围：0-999
   - 同一人员可能存在多个违规行为，请不要遗漏，输出全部违规类型**

**【输出格式】**
请以JSON格式输出检测结果：
```json
[
  {{
    "bbox_2d": [x1, y1, x2, y2],
    "violation_type": "类型"
  }}
]
```
**说明**：
- 如果一个工作人员同时存在多个违规（如玩手机且不穿工服），需要为每个违规行为单独输出一个bbox_2d
- 如果没有检测到违规行为，请返回空数组[]
"""

    completion = client.chat.completions.create(
        model=AI_MODEL,
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
    print("\n人员违规检测结果:")
    print("="*80)
    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
    print("="*80)

    return response_text


def plot_personnel_violations(image_path, personnel_json_path, output_path):
    """
    第五步（1）：在图片上绘制人员违规行为检测框（玩手机、睡觉、不穿工服）
    读取人员违规JSON文件并绘制

    参数:
        image_path: 原始图片路径
        personnel_json_path: 人员违规JSON文件路径
        output_path: 输出图片路径
    """
    print("\n" + "="*80)
    print("第五步（1）：绘制人员违规行为检测框")
    print("="*80)

    # 读取JSON文件
    try:
        with open(personnel_json_path, 'r', encoding='utf-8') as f:
            json_output = f.read()
        print(f"已读取人员违规JSON文件: {personnel_json_path}")
    except Exception as e:
        print(f"无法读取JSON文件 {personnel_json_path}: {e}")
        return False

    # 加载图像
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"无法打开图片: {image_path}")
        print(f"错误: {e}")
        return False
    width, height = img.size
    print(f"图像尺寸: {img.size}")

    draw = ImageDraw.Draw(img)

    # 定义人员违规类型的颜色和标签位置
    personnel_violation_config = {
        "玩手机": {
            "color": "red",
            "position": "top-left"  # 左上角
        },
        "便服": {
            "color": "yellow",
            "position": "bottom-left"  # 左下角
        },
        "不穿工服": {
            "color": "yellow",
            "position": "bottom-left"  # 左下角
        },
        "睡觉": {
            "color": "orange",
            "position": "top-right"  # 右上角
        }
    }

    # 解析JSON输出
    try:
        json_text = parse_json_output(json_output)
        violations = json.loads(json_text)

        if not isinstance(violations, list):
            violations = [violations]

        if len(violations) == 0:
            print("未检测到人员违规行为")

            # 保存原图
            img.save(output_path)
            print(f"结果已保存到: {output_path}")
            return True

        print(f"\n检测到 {len(violations)} 个人员违规行为")

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)  # Windows微软雅黑
            except:
                font = ImageFont.load_default()

        # 绘制每个人员违规检测框
        for i, violation in enumerate(violations):
            # 获取违规类型
            v_type = violation.get("violation_type", "未知违规")
            config = personnel_violation_config.get(v_type, {"color": "red", "position": "top-left"})
            color = config["color"]
            position = config["position"]

            # 获取bbox坐标
            bbox = violation.get("bbox_2d", [0, 0, 0, 0])
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

            print(f"\n人员违规 {i+1}:")
            print(f"  类型: {v_type}")
            print(f"  坐标(归一化): [{x1}, {y1}, {x2}, {y2}]")
            print(f"  坐标(实际): [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")

            # 绘制矩形框
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)),
                outline=color,
                width=4
            )

            # 根据违规类型确定标签位置
            label_text = f"{v_type}"

            # 获取文字大小以便调整位置
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # 根据配置位置确定标签坐标
            if position == "top-left":
                # 左上角
                text_x = abs_x1 + 8
                text_y = abs_y1 + 6
            elif position == "top-right":
                # 右上角
                text_x = abs_x2 - text_width - 8
                text_y = abs_y1 + 6
            elif position == "bottom-left":
                # 左下角
                text_x = abs_x1 + 8
                text_y = abs_y2 - text_height - 6
            elif position == "bottom-right":
                # 右下角
                text_x = abs_x2 - text_width - 8
                text_y = abs_y2 - text_height - 6
            else:
                # 默认左上角
                text_x = abs_x1 + 8
                text_y = abs_y1 + 6

            draw.text((text_x, text_y), label_text, fill=color, font=font)

        # 保存结果图像
        img.save(output_path)
        print(f"\n人员违规检测结果已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"解析或绘制失败: {e}")
        print(f"原始输出前200字: {json_output[:200]}")
        return False


def plot_workstation_violations(image_path, workstation_json_path, output_path):
    """
    第五步（2）：在图片上绘制工位违规检测框（岗位无人）
    读取工位违规JSON文件并绘制

    参数:
        image_path: 已绘制人员违规的图片路径
        workstation_json_path: 工位违规JSON文件路径
        output_path: 输出图片路径
    """
    print("\n" + "="*80)
    print("第五步（2）：绘制工位违规检测框")
    print("="*80)

    # 读取JSON文件
    try:
        with open(workstation_json_path, 'r', encoding='utf-8') as f:
            json_output = f.read()
        print(f"已读取工位违规JSON文件: {workstation_json_path}")
    except Exception as e:
        print(f"无法读取JSON文件 {workstation_json_path}: {e}")
        return False

    # 加载图像（注意：这里加载的是已经绘制过人员违规的图片）
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"无法打开图片: {image_path}")
        print(f"错误: {e}")
        return False
    width, height = img.size
    print(f"图像尺寸: {img.size}")

    draw = ImageDraw.Draw(img)

    # 定义工位违规类型的颜色和标签位置
    workstation_violation_config = {
        "岗位无人": {
            "color": "purple",
            "position": "bottom-right"  # 右下角
        }
    }

    # 解析JSON输出
    try:
        json_text = parse_json_output(json_output)
        violations = json.loads(json_text)

        if not isinstance(violations, list):
            violations = [violations]

        if len(violations) == 0:
            print("未检测到工位违规行为（无人工位）")

            # 保存图片（保持之前绘制的人员违规）
            img.save(output_path)
            print(f"结果已保存到: {output_path}")
            return True

        print(f"\n检测到 {len(violations)} 个工位违规行为（无人工位）")

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
        except:
            try:
                font = ImageFont.truetype("msyh.ttc", size=25)  # Windows微软雅黑
            except:
                font = ImageFont.load_default()

        # 绘制每个工位违规检测框
        for i, violation in enumerate(violations):
            # 获取违规类型
            v_type = violation.get("violation_type", "岗位无人")
            config = workstation_violation_config.get(v_type, {"color": "purple", "position": "bottom-right"})
            color = config["color"]
            position = config["position"]

            # 获取bbox坐标
            bbox = violation.get("bbox_2d", [0, 0, 0, 0])
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

            print(f"\n工位违规 {i+1}:")
            print(f"  类型: {v_type}")
            print(f"  坐标(归一化): [{x1}, {y1}, {x2}, {y2}]")
            print(f"  坐标(实际): [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")

            # 绘制矩形框
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)),
                outline=color,
                width=4
            )

            # 根据违规类型确定标签位置
            label_text = f"{v_type}"

            # 获取文字大小以便调整位置
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # 根据配置位置确定标签坐标
            if position == "top-left":
                # 左上角
                text_x = abs_x1 + 8
                text_y = abs_y1 + 6
            elif position == "top-right":
                # 右上角
                text_x = abs_x2 - text_width - 8
                text_y = abs_y1 + 6
            elif position == "bottom-left":
                # 左下角
                text_x = abs_x1 + 8
                text_y = abs_y2 - text_height - 6
            elif position == "bottom-right":
                # 右下角
                text_x = abs_x2 - text_width - 8
                text_y = abs_y2 - text_height - 6
            else:
                # 默认右下角
                text_x = abs_x2 - text_width - 8
                text_y = abs_y2 - text_height - 6

            draw.text((text_x, text_y), label_text, fill=color, font=font)

        # 保存结果图像
        img.save(output_path)
        print(f"\n工位违规检测结果已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"解析或绘制失败: {e}")
        print(f"原始输出前200字: {json_output[:200]}")
        return False


def process_folder(input_folder, output_folder, api_key, base_url):
    """
    处理文件夹中的所有图片
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_folder):
        if os.path.splitext(file)[1] in image_extensions:
            image_files.append(file)

    if not image_files:
        print(f"在 {input_folder} 中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 处理每张图片
    success_count = 0
    fail_count = 0

    for i, image_file in enumerate(image_files):
        print("\n" + "="*80)
        print(f"正在处理 [{i+1}/{len(image_files)}]: {image_file}")
        print("="*80)

        input_path = os.path.join(input_folder, image_file)

        # 构造输出文件名（统一使用小写扩展名）
        base_name = os.path.splitext(image_file)[0]
        ext = os.path.splitext(image_file)[1].lower()  # 转换为小写
        output_file = f"{base_name}_result{ext}"
        output_path = os.path.join(output_folder, output_file)

        # 构造工位和人员分析txt文件名
        workstation_txt = os.path.join(output_folder, f"{base_name}_workstation.txt")
        behavior_txt = os.path.join(output_folder, f"{base_name}_behavior.txt")

        try:
            # 编码图像
            print(f"正在读取图像...")
            print(f"完整路径: {input_path}")

            # 验证文件是否存在
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"文件不存在: {input_path}")

            # 获取文件大小
            file_size = os.path.getsize(input_path)
            print(f"文件大小: {file_size} 字节")

            image_base64, mime_type = encode_image(input_path)
            print(f"图像已编码: {len(image_base64)} 字符, 类型: {mime_type}")

            # 第一步：分析工位状态
            print("\n" + "█"*80)
            print("第一步：工位状态分析（描述工位情况）")
            print("█"*80)
            workstation_description = analyze_workstation(client, image_base64, input_path, workstation_txt)

            # 第二步：分析人员行为
            print("\n" + "█"*80)
            print("第二步：人员行为分析（描述人员行为状态）")
            print("█"*80)
            behavior_description = analyze_personnel_behavior(client, image_base64, input_path, behavior_txt)

            # 第三步：检测工位违规并保存JSON（基于工位状态描述）
            print("\n" + "█"*80)
            print("第三步：检测工位违规行为并标注坐标（基于工位状态分析）")
            print("█"*80)
            workstation_json_result = detect_workstation_violations(client, image_base64, input_path,
                                                                   workstation_description=workstation_description,
                                                                   mime_type=mime_type)
            # 保存工位违规JSON文件
            workstation_json_path = os.path.join(output_folder, f"{base_name}_workstation.json")
            with open(workstation_json_path, 'w', encoding='utf-8') as f:
                f.write(workstation_json_result)
            print(f"✓ 工位违规JSON已保存到: {workstation_json_path}")

            # 第三步半：简化人员描述（剔除顾客，只保留工作人员）
            print("\n" + "█"*80)
            print("第三步半：简化人员描述（提取工作人员信息）")
            print("█"*80)
            simplified_behavior_txt = os.path.join(output_folder, f"{base_name}_staff_simplified.txt")
            simplified_behavior_description = simplify_staff_description(client, behavior_description, simplified_behavior_txt)

            # 第四步：检测人员违规并保存JSON（基于简化后的人员行为描述）
            print("\n" + "█"*80)
            print("第四步：检测人员违规行为并标注坐标（基于简化后的人员行为分析）")
            print("█"*80)
            personnel_json_result = detect_personnel_violations(client, image_base64, input_path,
                                                               simplified_behavior_description=simplified_behavior_description,
                                                               mime_type=mime_type)
            # 保存人员违规JSON文件
            personnel_json_path = os.path.join(output_folder, f"{base_name}_personnel.json")
            with open(personnel_json_path, 'w', encoding='utf-8') as f:
                f.write(personnel_json_result)
            print(f"✓ 人员违规JSON已保存到: {personnel_json_path}")

            # 第五步（1）：绘制人员违规检测框（读取人员违规JSON）
            print("\n" + "█"*80)
            print("第五步（1）：绘制人员违规检测框（读取人员违规JSON）")
            print("█"*80)
            success_personnel = plot_personnel_violations(input_path, personnel_json_path, output_path)

            # 确认第五步（1）的输出文件已存在
            if not os.path.exists(output_path):
                print(f"✗ 错误：第五步（1）的输出文件未创建: {output_path}")
                success_workstation = False
            else:
                print(f"✓ 第五步（1）输出文件已创建: {output_path}")

                # 第五步（2）：绘制工位违规检测框（读取工位违规JSON）
                print("\n" + "█"*80)
                print("第五步（2）：绘制工位违规检测框（读取工位违规JSON）")
                print("█"*80)
                # 注意：这里读取的图片路径应该与第五步（1）输出的路径完全一致
                success_workstation = plot_workstation_violations(output_path, workstation_json_path, output_path)

            # 两个绘图步骤都成功才算成功
            success = success_personnel and success_workstation

            if success:
                success_count += 1
                print(f"✓ {image_file} 处理成功")
            else:
                fail_count += 1
                print(f"✗ {image_file} 处理失败")

        except FileNotFoundError as e:
            fail_count += 1
            print(f"✗ 文件未找到: {e}")
        except ValueError as e:
            fail_count += 1
            print(f"✗ 图片验证失败: {e}")
        except Exception as e:
            fail_count += 1
            import traceback
            print(f"✗ 处理 {image_file} 时出错: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())

    # 输出统计
    print("\n" + "="*80)
    print("处理完成！")
    print("="*80)
    print(f"总计: {len(image_files)} 张图片")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"输出文件夹: {output_folder}")
    print("="*80)


def main():
    """主函数"""
    # 配置
    input_folder = r"E:xxxx" #可以放一个父文件夹地址，父文件夹下可以是各个营业厅的子文件夹 
    api_key = "sk-xxxx"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    print("="*80)
    print("员工行为检测工具")
    print("检测类型：玩手机、睡觉、不穿工服、岗位无人")
    print("="*80)
    print(f"基础输入文件夹: {base_input_folder}")
    print("\n处理流程：")
    print("  第一步：工位状态分析（描述工位情况，生成 workstation.txt）")
    print("  第二步：人员行为分析（描述人员行为状态，生成 behavior.txt）")
    print("  第三步：检测工位违规（基于工位状态，生成 workstation.json）")
    print("  第三步半：简化人员描述（剔除顾客，只保留工作人员，生成 staff_simplified.txt）")
    print("  第四步：检测人员违规（基于简化后的人员行为，生成 personnel.json）")
    print("  第五步（1）：绘制人员违规检测框（读取 personnel.json）")
    print("  第五步（2）：绘制工位违规检测框（读取 workstation.json）")
    print("="*80)

    # 获取所有子文件夹
    try:
        subfolders = []
        for item in os.listdir(base_input_folder):
            item_path = os.path.join(base_input_folder, item)
            if os.path.isdir(item_path):
                subfolders.append(item_path)

        if not subfolders:
            print(f"\n在 {base_input_folder} 中没有找到子文件夹")
            return

        print(f"\n找到 {len(subfolders)} 个子文件夹：")
        for i, folder in enumerate(subfolders, 1):
            folder_name = os.path.basename(folder)
            print(f"  {i}. {folder_name}")
        print("="*80)

    except Exception as e:
        print(f"读取文件夹失败: {e}")
        return

    # 处理每个子文件夹
    total_success = 0
    total_fail = 0

    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        output_folder = subfolder + "_result"

        print("\n" + "╔" + "="*78 + "╗")
        print(f"║  正在处理子文件夹: {folder_name}")
        print("╚" + "="*78 + "╝")

        try:
            # 处理当前子文件夹
            process_folder(subfolder, output_folder, api_key, base_url)
            total_success += 1
        except Exception as e:
            total_fail += 1
            print(f"\n✗ 处理子文件夹 {folder_name} 时出错: {e}")
            import traceback
            print(traceback.format_exc())

    # 输出总体统计
    print("\n" + "="*80)
    print("所有子文件夹处理完成！")
    print("="*80)
    print(f"总计: {len(subfolders)} 个子文件夹")
    print(f"成功: {total_success} 个")
    print(f"失败: {total_fail} 个")
    print("="*80)


if __name__ == "__main__":
    main()
