from traitlets.utils.descriptions import describe

tools = [
    {
        "type":"function",
        "function":{
            "name":"query_by_product_name",
            "description": "Query the database to retrieve a list of products that match or contain the specified product name. This function can be used to assist customers in finding products by name via an online platform or customer support interface.",
            "parameters":{
                "type":"object",
                "properties":{
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to search for. The search is case-insensitive and allows partial matches."
                    }
                }
            },
            "required":["product_name"],
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_store_promotions",
            "description": "Read the store's promotion document to find specific promotions related to the provided product name. This function scans a text document for any promotional entries that include the product name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to search for in the promotion document. The function returns the promotional details if found."
                    }
                },
                "required": ["product_name"]
            }
        }
    }
]