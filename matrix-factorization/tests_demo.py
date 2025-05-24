def get_tests():
    return (
        {
            "name": "remove_hierarchical_cluster",
            "icons_present": (
                "Data-File",
                "Data-Data Table",
                "Unsupervised-Distances",
            ),
            "icons_removed": (
                "Unsupervised-Hierarchical Clustering",
            ),
        },
        {
            "name": "remove_box_plot",
            "icons_present": (
                "Data-File",
                "Data-Data Table",
                "Data-Data Table",
                "Unsupervised-Distances",
                "Unsupervised-Hierarchical Clustering",
            ),
            "icons_removed": (
                "Visualize-Box Plot",
            ),
        },
        {
            "name": "remove_forest",
            "icons_present": (
                "Data-File",
                "Data-Data Table",
                "Data-Data Table",
                "Model-Logistic Regression",
                "Evaluate-Test and Score",
                "Evaluate-Confusion Matrix",
            ),
            "icons_removed": (
                "Model-Random Forest",
            ),
        },
        {
            "name": "remove_confusion",
            "icons_present": (
                "Data-File",
                "Data-Data Table",
                "Data-Data Table",
                "Model-Logistic Regression",
                "Evaluate-Test and Score",
                "Model-Random Forest",
            ),
            "icons_removed": (
                "Evaluate-Confusion Matrix",
            ),
        },
        {
            "name": "remove_scatter",
            "icons_present": (
                "Data-File",
                "Evaluate-Test and Score",
                "Evaluate-Confusion Matrix",
                "Model-Logistic Regression",
            ),
            "icons_removed": (
                "Visualize-Scatter Plot",
            ),
        },
        {
            "name": "remove_confusion_2",
            "icons_present": (
                "Data-File",
                "Evaluate-Test and Score",
                "Model-Logistic Regression",
            ),
            "icons_removed": (
                "Evaluate-Confusion Matrix",
            ),
        },
        {
            "name": "remove_box",
            "icons_present": (
                "Data-Datasets",
                "Survival Analysis-Kaplan-Meier Plot",
                "Data-Data Table",
            ),
            "icons_removed": (
                "Visualize-Box Plot",
            ),
        },
        {
            "name": "remove_kaplan_meier",
            "icons_present": (
                "Data-Datasets",
                "Visualize-Distributions",
                "Survival Analysis-Rank Survival Features",
            ),
            "icons_removed": (
                "Survival Analysis-Kaplan-Meier Plot",
            ),
        },#my tests start
        {
            "name": "remove-logistic-regression",
            "icons_present": (
                "Data-File",
                "Evaluate-Test and Score",
                "Transform-Data Sampler",
            ),
            "icons_removed": (
                "Model-Logistic Regression",
            ),
        },
        {
            "name": "remove-nomogram",
            "icons_present": (
                "Text Mining-Corpus",
                "Text Mining-Preprocess Text",
                "Text Mining-Bag of Words",
            ),
            "icons_removed": (
                "Visualize-Nomogram",
            ),
        },
        {
            "name": "remove-tree",
            "icons_present": (
                "Data-File",
                "Visualize-Tree Viewer",
                "Visualize-Scatter Plot",
                "Visualize-Box Plot",
            ),
            "icons_removed": (
                "Model-Tree",
            ),
        },
        {
            "name": "remove-scatter",
            "icons_present": (
                "Data-File",
                "Data-Data Table",
                "Unsupervised-PCA",
            ),
            "icons_removed": (
                "Visualize-Scatter Plot",
            ),
        }
        
        
    )