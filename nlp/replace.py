def replace_named_entities(query, results, men_list, woman_list):
    # Sắp xếp kết quả theo vị trí bắt đầu để tránh xáo trộn chỉ số khi thay thế
    results = sorted(results, key=lambda x: x["start"], reverse=True)
    processed_words = set()  # Lưu trữ các từ đã thay thế

    for result in results:
        word = result["word"]
        start = result["start"]
        end = result["end"]

        if word in men_list:
            if word in processed_words:
                replacement = "người đàn ông"  # Giữ nguyên nếu đã xử lý trước đó
            else:
                replacement = "người đàn ông khác"
                processed_words.add(word)

            query = query[:start] + replacement + query[end:]

        elif word in woman_list:
            if word in processed_words:
                replacement = "người phụ nữ"  # Giữ nguyên nếu đã xử lý trước đó
            else:
                replacement = "người phụ nữ khác"
                processed_words.add(word)

            query = query[:start] + replacement + query[end:]

    return query
