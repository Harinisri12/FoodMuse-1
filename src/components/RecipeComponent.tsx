import React, { useState, useEffect } from "react";
import { Trash2 } from "lucide-react";
import { getSavedRecipesFromDB, deleteRecipeFromDB } from "./indexedDB";

const SavedRecipes: React.FC = () => {
    const [savedRecipes, setSavedRecipes] = useState<any[]>([]);

    useEffect(() => {
        async function fetchRecipes() {
            const recipes = await getSavedRecipesFromDB();
            setSavedRecipes(recipes);
        }
        fetchRecipes();
    }, []);

    const removeRecipe = async (recipeId: number) => {
        await deleteRecipeFromDB(recipeId);
        setSavedRecipes(savedRecipes.filter((recipe) => recipe.id !== recipeId));
    };

    const openRecipeDetails = (recipe: any) => {
        const newTab = window.open("", "_blank");
        if (newTab) {
            newTab.document.write(`
        <html>
          <head>
            <title>${recipe.title}</title>
            <style>
              body { font-family: Arial, sans-serif; padding: 30px; background: #f5f5f5; }
              h1 { text-align: center; color: #047857; }
              .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; }
              img { width: 100%; border-radius: 10px; margin-bottom: 20px; }
              h2 { color: #047857; margin-top: 20px; }
              ul { list-style: none; padding: 0; }
              li { margin-bottom: 10px; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>${recipe.title}</h1>
              <img src="${recipe.image}" alt="${recipe.title}" />
              <h2>Ingredients:</h2>
              <ul>${recipe.extendedIngredients.map((i: any) => `<li>• ${i.original}</li>`).join("")}</ul>
              <h2>Instructions:</h2>
              <p>${recipe.instructions || "No instructions available."}</p>
            </div>
          </body>
        </html>
      `);
            newTab.document.close();
        }
    };

    if (savedRecipes.length === 0) {
        return (
            <div className="min-h-screen bg-gray-50 p-8 text-center">
                <h1 className="text-3xl font-bold text-gray-800 mb-4">Saved Recipes</h1>
                <p className="text-gray-600">You haven't saved any recipes yet.</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-6">Saved Recipes</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {savedRecipes.map((recipe) => (
                    <div key={recipe.id} className="bg-white rounded-lg shadow-md overflow-hidden">
                        <img src={recipe.image} alt={recipe.title} className="w-full h-48 object-cover" />
                        <div className="p-4">
                            <h2 className="text-xl font-semibold mb-2">{recipe.title}</h2>
                            <div className="flex justify-between items-center mt-4">
                                <button onClick={() => openRecipeDetails(recipe)} className="text-emerald-600 hover:text-emerald-800 transition-colors">
                                    View Details
                                </button>
                                <button onClick={() => removeRecipe(recipe.id)} className="text-red-500 hover:text-red-700 transition-colors">
                                    <Trash2 className="w-5 h-5" />
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SavedRecipes;
