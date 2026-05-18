from app.services.recommendation.preference_extractor import PreferenceExtractorService


def main():
    extractor = PreferenceExtractorService()

    
    result = extractor.extract(
            user_message="อยากได้รถขี่ในเมือง...",
            schema_type="item_based",
    )

    print("MODEL:", result.model_name)
    print("SCHEMA:", result.schema_type)
    print("PREFERENCE:")
    print(result.preference)


if __name__ == "__main__":
    main()