#!/usr/bin/env python3
"""Fix font settings in full_analysis.py to use English fallback"""
import re

with open('/root/mtl4rec/full_analysis.py', 'r') as f:
    content = f.read()

# Replace font family
old_font = "plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']"
new_font = "plt.rcParams['font.family'] = ['DejaVu Sans']"
content = content.replace(old_font, new_font)

# Replace Chinese task names
old_cn = """TASK_NAMES_CN = {
    'is_click': '点击', 'is_like': '点赞', 'is_follow': '关注',
    'is_comment': '评论', 'is_forward': '转发', 'long_view': '长时观看',
    'is_hate': '负反馈', 'is_profile_enter': '进主页'
}"""
new_en = """TASK_NAMES_CN = {
    'is_click': 'Click', 'is_like': 'Like', 'is_follow': 'Follow',
    'is_comment': 'Comment', 'is_forward': 'Forward', 'long_view': 'LongView',
    'is_hate': 'Hate', 'is_profile_enter': 'Profile'
}"""
content = content.replace(old_cn, new_en)

# Also fix identify_model to handle v5_screen.log and v5v2
old_identify = """def identify_model(log_name):
    \"\"\"从日志文件名识别模型和版本\"\"\"
    name = log_name.replace('.log', '')
    m = re.match(r'(PLE_v?\d+|MMoE_v?\d+|SharedBottom|PLE|MMoE)_KuaiRand1K_(v?\d+[a-z]*)?_?(\d{8}_\d{6})', name)
    if m:
        model_name = m.group(1)
        version = m.group(2) if m.group(2) else 'base'
        return model_name, version
    m = re.match(r'(SharedBottom|MMoE|PLE)_KuaiRand1K_(\d{8}_\d{6})', name)
    if m:
        return m.group(1), 'base'
    return name, 'unknown'"""

new_identify = """def identify_model(log_name):
    \"\"\"从日志文件名识别模型和版本\"\"\"
    name = log_name.replace('.log', '')
    
    # Handle special cases
    if 'v5_screen' in name:
        return 'PLE', 'v5'
    if 'v5v2' in name:
        return 'MMoE', 'v2'
    
    m = re.match(r'(PLE_v?\d+|MMoE_v?\d+|SharedBottom|PLE|MMoE)_KuaiRand1K_(v?\d+[a-z]*)?_?(\d{8}_\d{6})', name)
    if m:
        model_name = m.group(1)
        version = m.group(2) if m.group(2) else 'base'
        return model_name, version
    m = re.match(r'(SharedBottom|MMoE|PLE)_KuaiRand1K_(\d{8}_\d{6})', name)
    if m:
        return m.group(1), 'base'
    return name, 'unknown'"""

content = content.replace(old_identify, new_identify)

with open('/root/mtl4rec/full_analysis.py', 'w') as f:
    f.write(content)

print("Font and identify_model updated successfully")
print(f"File size: {len(content)} bytes")