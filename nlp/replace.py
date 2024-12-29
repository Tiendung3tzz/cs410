def replace_named_entities(query, results, men_list, woman_list):
    # Sắp xếp kết quả theo vị trí bắt đầu để tránh xáo trộn chỉ số khi thay thế
    processed_men = set()  # Lưu trữ các từ đã thay thế cho đàn ông
    processed_woman = set()  # Lưu trữ các từ đã thay thế cho phụ nữ

    offset = 0  # Độ lệch để cập nhật chỉ số start và end

    for result in results:
        word = result["word"]
        start = result["start"] + offset  # Áp dụng độ lệch
        end = result["end"] + offset  # Áp dụng độ lệch
        print(word)
        if word in men_list:
            if word not in processed_men and len(processed_men) > 0:
                replacement = "người đàn ông khác"
                processed_men.add(word)
            else:
                replacement = "người đàn ông"
                processed_men.add(word)

        elif word in woman_list:
            if word not in processed_woman and len(processed_woman) > 0:
                replacement = "người phụ nữ khác"
                processed_woman.add(word)
            else:
                replacement = "người phụ nữ"
                processed_woman.add(word)
        else:
            continue  # Nếu từ không nằm trong danh sách, bỏ qua

        # Thay thế từ trong query và tính toán độ lệch
        query = query[:start] + replacement + query[end:]
        offset += len(replacement) - (end - start)  # Cập nhật độ lệch

    return query
