from excel_actions import handle_intent

print("SUM of Sales:", handle_intent("SUM", column="Sales"))
print("AVERAGE of Profit:", handle_intent("AVERAGE", column="Profit"))
print("MAX of Profit:", handle_intent("FIND_MAX", column="Profit"))
print("FILTER by Region == 'East':", handle_intent("FILTER", condition='Region == "East"'))
print("Simulated chart:", handle_intent("CREATE_CHART", column="Sales"))
