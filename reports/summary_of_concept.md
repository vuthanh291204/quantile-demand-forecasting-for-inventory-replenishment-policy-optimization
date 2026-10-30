# Tóm tắt khái niệm quản lý tồn kho

## 1. Kho phải quyết định điều gì?

Mỗi ngày kho phải trả lời hai câu: khi nào đặt hàng và đặt bao nhiêu. Đặt quá nhiều thì hàng nằm trong kho, tốn chi phí lưu kho. Đặt quá ít thì hết hàng, khách không mua được. Các khái niệm dưới đây là những công cụ để cân bằng hai điều này.

## 2. Thời gian chờ hàng (lead time)

Là số ngày từ lúc đặt hàng đến lúc hàng về kho. Vì hàng không về ngay, kho phải đặt hàng trước khi hết. Dữ liệu M5 không có thông tin này, nên trong đồ án đây là một tham số giả định (dự kiến thử 3, 7, 14 ngày).

## 3. Nhu cầu trong thời gian chờ

Là tổng số hàng bán ra trong những ngày chờ hàng. Ví dụ minh họa: trung bình bán 2 đơn vị mỗi ngày, thời gian chờ 4 ngày, thì nhu cầu dự kiến trong thời gian chờ là 2 × 4 = 8 đơn vị. Đây là con số kho cần "phủ" trong lúc hàng chưa về.

## 4. Tồn kho an toàn (safety stock)

Có hai cách tính thường gặp, ứng với hai nguồn bất định khác nhau. (a) Khi thời gian giao hàng không chắc chắn: tồn kho an toàn = (tiêu thụ trung bình mỗi ngày × thời gian giao tối đa) − (tiêu thụ trung bình mỗi ngày × thời gian giao trung bình). (b) Khi nhu cầu không chắc chắn: tồn kho an toàn = z × độ lệch chuẩn của nhu cầu trong thời gian chờ, với z ứng với mức đáp ứng mong muốn. Trong đồ án, thời gian chờ hàng được giả định cố định nên công thức (a) cho kết quả 0; nguồn bất định chính là nhu cầu, nên cách (b) phù hợp hơn. Cách (b) thường giả định nhu cầu gần với phân phối chuẩn, giả định này có thể không khớp với chuỗi có nhiều ngày bán 0.

## 5. Điểm đặt hàng (reorder point)

Là mức tồn kho mà khi chạm tới thì phải đặt thêm hàng:

> điểm đặt hàng = nhu cầu dự kiến trong thời gian chờ + tồn kho an toàn

Ví dụ minh họa: 8 + 4 = 12 đơn vị. Khi so với mức này, cần tính cả hàng đang trên đường về, không chỉ hàng đang có trong kho; nếu không kho sẽ đặt trùng. Hai cách đặt hàng khác trong đồ án là (s,S) (khi tồn kho tụt xuống s thì đặt để đưa lên mức S) và đặt theo dự báo theo khoảng.

## 6. Mức đáp ứng đơn (service level)

Trong đồ án này, mức đáp ứng đơn là tỉ lệ nhu cầu của khách mà kho đáp ứng được: 95% nghĩa là cứ 100 lượt muốn mua thì 95 lượt có hàng.

## 7. Dự báo theo khoảng (quantile) và liên hệ với điểm đặt hàng

Dự báo phân vị 90% là con số X sao cho có khoảng 90% khả năng nhu cầu không vượt quá X. Ý tưởng của đồ án: lấy phân vị của nhu cầu trong cả thời gian chờ làm cơ sở đặt hàng, thay vì tự cộng tồn kho an toàn theo công thức thống kê. Có một lưu ý: cần phân vị của tổng nhu cầu cả giai đoạn chờ; cộng các phân vị theo từng ngày không cho cùng kết quả. Cách tính đúng sẽ chọn ở các tuần sau.

## 8. Các khái niệm gắn với câu hỏi nghiên cứu như thế nào?

Câu hỏi là: ở cùng mức đáp ứng đơn, đặt hàng theo dự báo theo khoảng có tốn ít chi phí lưu kho hơn cách dùng thống kê lịch sử không? Muốn so sánh công bằng, mỗi cách đặt hàng phải được chỉnh tham số sao cho mức đáp ứng đơn đạt được trong mô phỏng bằng nhau (ví dụ cùng 95%). Chỉ chọn "phân vị 95%" chưa chắc cho đúng mức đáp ứng 95%, nên cần kiểm tra bằng mô phỏng.
