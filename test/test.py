
mydict = {
    "u": {
        "order": 2,
        "dim": "vector"
    },
        "m": {
        "order": 2,
        "dim": "skalar"
    },
    "p": {
        "order": 1,
        "dim": "skalar"
    }
}

# Filtere das Dictionary, um nur die Elemente mit "dim" gleich "skalar" zu behalten
filtered_dict = {key: value for key, value in mydict.items() if value.get("dim") == "skalar"}

print(filtered_dict)


print(str(["fe1", "fe_2"]))