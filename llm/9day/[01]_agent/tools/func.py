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
    }
]