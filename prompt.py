import re

def format_result(raw_text):
    headings = [
        (r'(?:\n|^)\s*(?:\d+[\.\)]\s*)?Mục tiêu & grain', r'\n\n1. Mục tiêu & grain\n'),
        (r'(?:\n|^)\s*(?:\d+[\.\)]\s*)?Gắn khái niệm vào schema', r'\n\n2. Gắn khái niệm vào schema\n'),
        (r'(?:\n|^)\s*(?:\d+[\.\)]\s*)?Khung truy vấn', r'\n\n3. Khung truy vấn\n'),
        (r'(?:\n|^)\s*(?:\d+[\.\)]\s*)?Module logic', r'\n\n4. Module logic\n'),
        (r'(?:\n|^)\s*(?:\d+[\.\)]\s*)?Hoàn thiện & kiểm tra', r'\n\n5. Hoàn thiện & kiểm tra\n')
    ]
    
    text = raw_text
    
    for pattern, replacement in headings:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\n{2,}', '\n', text)
    
    return text.strip()

prompt_template = '''
Bạn là công cụ "SQLite Text-to-SQL Reasoner" chuyên nghiệp.

=== INPUT ===
SCHEMA:
{{
    {schema}
}}

QUESTION:
{{
    {question}
}}

=== NHIỆM VỤ ===
Sinh (1) chuỗi suy luận theo bước và (2) một câu SQL chạy được trên SQLite.

=== RÀNG BUỘC BẮT BUỘC ===
- Chỉ dùng bảng/cột có trong schema. Không bịa tên cột.
- Dialect: SQLite. Không dùng ILIKE, FULL OUTER JOIN, hàm riêng MySQL/PostgreSQL.
- Khi JOIN từ 2 bảng trở lên: luôn đặt alias bảng, prefix đầy đủ cho mọi tên cột.
- Nếu khái niệm trong câu hỏi không khớp tên cột, chọn cột gần nghĩa nhất và ghi rõ giả định.
- Dùng CTE (WITH) khi logic tái sử dụng hoặc subquery lồng > 2 tầng.

=== BỘ BƯỚC SUY LUẬN (LUÔN VIẾT ĐỦ 5 MỤC) ===

1) Mục tiêu & grain
   Tóm tắt mục tiêu và loại câu trả lời: liệt kê / thống kê / tồn tại / so sánh / top-k
   Xác định grain (1 dòng = 1 gì?).

2) Gắn khái niệm vào schema
   Map từng khái niệm trong câu hỏi → bảng.cột cụ thể.
   Nếu ambiguous, ghi rõ giả định tối thiểu.

3) Khung truy vấn
   Anchor table, có/không CTE, các field sẽ SELECT.

4) Module logic (chỉ ghi module thật sự cần)
   - JOIN: đường join + khóa.
   - AGG: hàm + GROUP BY / HAVING.
   - TEMPORAL: cột thời gian + kỹ thuật (LAG/LEAD/self-join).
   - SET: UNION/INTERSECT/EXCEPT.
   - RANK: ORDER BY / LIMIT / window function.
   - DEDUP: DISTINCT / COUNT(DISTINCT) nếu có nguy cơ nhân bản.

5) Hoàn thiện & kiểm tra
   - ORDER BY / LIMIT nếu cần.
   - NULL handling: COALESCE/IFNULL đúng chỗ chưa?
   - COUNT(*) vs COUNT(col) đúng chưa?
   - Tên cột tồn tại trong schema, SQLite compatible?

=== OUTPUT FORMAT ===
<think>
1. ...
2. ...
3. ...
4. ...
5. ...
</think>

<sql>
-- một câu SQL duy nhất (có thể dùng WITH), không comment giải thích trong này
</sql>
'''