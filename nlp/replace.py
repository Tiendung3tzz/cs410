
def replace_named_entities(query, results, men_list, woman_list):
    # Sắp xếp kết quả theo vị trí bắt đầu để tránh xáo trộn chỉ số khi thay thế
    results = sorted(results, key=lambda x: x["start"], reverse=True)
    for result in results:
        word = result["word"]
        start = result["start"]
        end = result["end"]
        if word in men_list:
            query = query[:start] + "người đàn ông" + query[end:]
        elif word in woman_list:
            query = query[:start] + "người phụ nữ" + query[end:]
    return query