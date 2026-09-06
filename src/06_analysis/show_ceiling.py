rows = [
    ("Metric", "Value"),
    ("Total training examples", "12,924"),
    ("Distinct feature vectors", "1,362"),
    ("Examples sharing vector w/ opposite label", "11,256 (87.1%)"),
    ("Theoretical max accuracy (majority vote)", "69.4%"),
    ("Largest duplicate group size", "1,400 examples"),
    ("Programs spanned by that one group", "263"),
    ("True correlation range in that group", "-1.0 to +1.0"),
]
for row in rows:
    print(f"{row[0]:<45}{row[1]:<20}")
    if row[0] == "Metric":
        print("-" * 65)
