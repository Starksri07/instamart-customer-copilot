-- 1. Orders grouped by status
SELECT LOWER(status) AS status,
       COUNT(*) AS order_rows
FROM orders
GROUP BY LOWER(status)
ORDER BY order_rows DESC;


-- 2. Top 10 products by quantity sold
SELECT p.product_name,
       SUM(o.quantity) AS units_sold,
       COUNT(*) AS order_rows
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE LOWER(o.status) = 'completed'
GROUP BY p.product_id, p.product_name
ORDER BY units_sold DESC
LIMIT 10;


-- 3. Customer count by city
SELECT UPPER(city) AS city,
       COUNT(*) AS customers
FROM customers
GROUP BY UPPER(city)
ORDER BY customers DESC;


-- 4. Completed sales by city
SELECT UPPER(c.city) AS city,
       ROUND(SUM(o.order_amount), 2) AS sales_value,
       COUNT(*) AS completed_order_rows
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE LOWER(o.status) = 'completed'
GROUP BY UPPER(c.city)
ORDER BY sales_value DESC;


-- 5. Top 20 customers by completed purchase value
SELECT c.customer_id,
       UPPER(c.city) AS city,
       COUNT(*) AS completed_order_rows,
       ROUND(SUM(o.order_amount), 2) AS total_spend
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE LOWER(o.status) = 'completed'
GROUP BY c.customer_id, UPPER(c.city)
ORDER BY total_spend DESC
LIMIT 20;


-- 6. Completed sales by product category
SELECT p.category,
       SUM(o.quantity) AS units_sold,
       ROUND(SUM(o.order_amount), 2) AS total_sales
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE LOWER(o.status) = 'completed'
GROUP BY p.category
ORDER BY total_sales DESC;


-- 7. Support tickets grouped by category
SELECT true_category AS ticket_category,
       COUNT(*) AS ticket_count
FROM tickets
GROUP BY true_category
ORDER BY ticket_count DESC;
