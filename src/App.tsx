/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from "axios";
import { Plus, X } from "lucide-react";
import { Navbar } from "./components/Navbar";
import { UserProvider, useUser } from "./context/UserContext";
import RecipeComponent from './components/RecipeComponent';
import { Login } from './components/Login';
import { Register } from './components/Register';
import { saveRecipeToDB, getSavedRecipesFromDB } from './components/indexedDB';
 const API_KEY = "9278ceb7678a45a5addf48b39c93364f";
//const API_KEY = "37e38fd0fd554ee19bfd50c08f600f4d"
//const API_KEY = "ce57e19380804d16aec9f5a2674c729f";

function IngredientInput({ onGenerateRecipes }: { onGenerateRecipes: (ingredients: string[]) => void }) {
  const [ingredients, setIngredients] = useState<string[]>([]);
  const [currentIngredient, setCurrentIngredient] = useState("");
  const addIngredient = () => {
    if (currentIngredient.trim()) {
      setIngredients([...ingredients, currentIngredient.trim()]);
      setCurrentIngredient("");
    }
  };

  const removeIngredient = (index: number) => {
    setIngredients(ingredients.filter((_, i) => i !== index));
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={currentIngredient}
          onChange={(e) => setCurrentIngredient(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addIngredient()}
          placeholder="Enter an ingredient..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
        />
        <button
          onClick={addIngredient}
          className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 flex items-center gap-2"
        >
          <Plus className="w-5 h-5" /> Add
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {ingredients.map((ingredient, index) => (
          <div key={index} className="flex items-center gap-2 px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full">
            <span>{ingredient}</span>
            <button onClick={() => removeIngredient(index)} className="hover:text-emerald-600">
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>

      <button onClick={() => onGenerateRecipes(ingredients)} className="px-6 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700">
        Generate Recipes
      </button>
    </div>
  );
}

function FilterCard({
  onFilterDuration,
  onFilterDishTypes,
  onFilterCuisine
}: {
  onFilterDuration: (duration: string) => void;
  onFilterDishTypes: (selectedDishTypes: string[]) => void;
  onFilterCuisine: (selectedCuisine: string) => void;
}) {
  const [selectedDishTypes, setSelectedDishTypes] = useState<string[]>([]);
  const [selectedCuisine, setSelectedCuisine] = useState<string>("");

  const dishTypes = [
    "main course", "side dish", "dessert", "appetizer", "salad", "bread",
    "breakfast", "soup", "beverage", "sauce", "marinade", "fingerfood",
    "snack", "drink"
  ];

  const cuisines = [
    "African", "Asian", "American", "British", "Cajun", "Caribbean",
    "Chinese", "Eastern European", "European", "French", "German", "Greek",
    "Indian", "Irish", "Italian", "Japanese", "Jewish", "Korean", "Latin American",
    "Mediterranean", "Mexican", "Middle Eastern", "Nordic", "Southern", "Spanish",
    "Thai", "Vietnamese"
  ];

  const handleDishTypeChange = (dishType: string) => {
    setSelectedDishTypes((prev) => {
      if (prev.includes(dishType)) {
        return prev.filter((item) => item !== dishType); // Remove if already selected
      } else {
        return [...prev, dishType]; // Add if not selected
      }
    });
  };

  const handleCuisineChange = (cuisine: string) => {
    setSelectedCuisine(cuisine);
    onFilterCuisine(cuisine);
  };

  const handleApplyFilter = () => {
    onFilterDishTypes(selectedDishTypes);
  };


  return (
    <div className="bg-white p-8 mb-6 w-full mx-auto">
      <h2 className="text-2xl font-semibold mb-4">Filter Recipes</h2>

      <div className="flex justify-between mb-6 w-full">
        {/* Left side: Duration Dropdown */}
        <div className="w-1/2 pr-4">
          <p className="text-sm text-gray-600 mb-4">Choose your preferred cooking duration:</p>
          <select
            onChange={(e) => onFilterDuration(e.target.value)}
            className="w-full px-5 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all text-gray-700"
          >
            <option value="">Select Duration</option>
            <option value="under20">Under 20 minutes</option>
            <option value="above20">Above 20 minutes</option>
          </select>
        </div>

        {/* Right side: Cuisine Dropdown */}
        <div className="w-1/2 pl-4">
          <p className="text-sm text-gray-600 mb-4">Select Cuisine:</p>
          <select
            value={selectedCuisine}
            onChange={(e) => handleCuisineChange(e.target.value)}
            className="w-full px-5 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 transition-all text-gray-700"
          >
            <option value="">Select Cuisine</option>
            {cuisines.map((cuisine) => (
              <option key={cuisine} value={cuisine}>{cuisine}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Dish Type Section */}
      <div className="mt-6 w-full">
        <p className="text-sm text-gray-600 mb-2">Select Dish Types:</p>
        <div className="flex flex-wrap gap-6 mb-6 w-full">
          {dishTypes.map((dishType) => (
            <div key={dishType} className="flex items-center">
              <input
                type="checkbox"
                id={dishType}
                checked={selectedDishTypes.includes(dishType)}
                onChange={() => handleDishTypeChange(dishType)}
                className="h-5 w-5 text-emerald-600 border-gray-300 rounded-md focus:ring-2 focus:ring-emerald-500 transition-all"
              />
              <label htmlFor={dishType} className="ml-3 text-gray-700">{dishType}</label>
            </div>
          ))}
        </div>
      </div>

      {/* Buttons */}
      <div className="mt-6 text-center w-full">
        <button
          onClick={handleApplyFilter}
          className="px-6 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
        >
          Apply Filters
        </button>
        <button
          onClick={() => { setSelectedDishTypes([]); setSelectedCuisine(""); onFilterDishTypes([]); onFilterCuisine(""); }}
          className="ml-4 text-sm text-emerald-600 hover:text-emerald-800 focus:outline-none transition-colors"
        >
          Reset Filter
        </button>
      </div>
    </div>


  );
}


function AppContent() {
  const { user, setUser, tempRegistrationData, setTempRegistrationData } = useUser();
  const [isRegistering, setIsRegistering] = useState(true);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filteredRecipes, setFilteredRecipes] = useState<any[]>([]);
  const [selectedDishTypes, setSelectedDishTypes] = useState<string[]>([]);
  const [selectedCuisine, setSelectedCuisine] = useState<string>("");
  const [recommendedRecipes, setRecommendedRecipes] = useState<any[]>([]);
  const [savedRecipes, setSavedRecipes] = useState<any[]>([]);

  useEffect(() => {
    // Load recipes from IndexedDB when the component mounts
    async function fetchRecipes() {
      const recipes = await getSavedRecipesFromDB();
      setSavedRecipes(recipes);
    }
    fetchRecipes();
  }, []);

  useEffect(() => {
    (window as any).saveRecipe = async (recipe: any) => {

      const isRecipeSaved = savedRecipes.some(savedRecipe => savedRecipe.id === recipe.id);

      if (!isRecipeSaved) {
        const updatedRecipes = [...savedRecipes, recipe];
        setSavedRecipes(updatedRecipes);

        await saveRecipeToDB(recipe);
      } else {
        console.log('Recipe already saved:', recipe.title);
      }
    };

    return () => {
      delete (window as any).saveRecipe;
    };
  }, [savedRecipes]);

  const handleLogin = (email: string) => {
    if (tempRegistrationData && tempRegistrationData.email === email) {
      setUser(tempRegistrationData);
    } else {
      const username = email.split('@')[0];
      setUser({ username, email, fullName: username });
    }
  };

  const handleRegister = (username: string, email: string, fullName: string) => {
    const userData = { username, email, fullName };
    setTempRegistrationData(userData);
    setIsRegistering(false); // Automatically switch to login
  };

  if (!user) {
    return isRegistering ? (
      <Register
        onRegister={handleRegister}
        onSwitchToLogin={() => setIsRegistering(false)}
      />
    ) : (
      <Login
        onLogin={handleLogin}
        onSwitchToRegister={() => setIsRegistering(true)}
      />
    );
  }

  const fetchRecipes = async (ingredients: string[]) => {
    if (ingredients.length === 0) {
      setError("Please enter at least one ingredient.");
      return;
    }
    setError(null);
    setLoading(true);
    setRecipes([]);
  
    console.log("🔹 Fetching recipes for:", ingredients);
  
    try {
      const response = await axios.get("https://api.spoonacular.com/recipes/findByIngredients", {
        params: { ingredients: ingredients.join(","), apiKey: API_KEY },
      });
  
      console.log("✅ Spoonacular API Response:", response.data);
  
      const recipePromises = response.data.map(async (recipe: any) => {
        const nutrition = await axios.get(`https://api.spoonacular.com/recipes/${recipe.id}/nutritionWidget.json`, {
          params: { apiKey: API_KEY },
        });
  
        const detailedRecipe = await axios.get(`https://api.spoonacular.com/recipes/${recipe.id}/information`, {
          params: { apiKey: API_KEY },
        });
  
        console.log(detailedRecipe.data); // Debugging line to check API response
  
        return {
          ...recipe,
          healthScore: detailedRecipe.data.healthScore, // ✅ Fixed key
          nutrition: nutrition.data,
          cookingTime: detailedRecipe.data.cookingMinutes || detailedRecipe.data.readyInMinutes,
          dishTypes: detailedRecipe.data.dishTypes || [],
          cuisines: detailedRecipe.data.cuisines || [],
        };
      });
  
      const recipesWithNutrition = await Promise.all(recipePromises);
  
      // 🔥 Send processed recipes to Flask backend
      const userProfile = { heart_disease: false, diabetes: false, high_bp: true, fitness_goal: "weight_loss" };
      const requestData = { userProfile, recipes: recipesWithNutrition };
  
      console.log("🔹 Sending data to /recommend:", requestData);
  
      const recommendationResponse = await axios.post("http://127.0.0.1:5000/recommend", requestData);
  
      console.log("✅ Flask Response:", recommendationResponse.data);
  
      // ✅ Update state with recommended recipes
      setRecommendedRecipes(recommendationResponse.data.recipes);
  
      setRecipes(recipesWithNutrition);
      setFilteredRecipes(recipesWithNutrition);
    } catch (err) {
      setError("Failed to fetch recipes. Please try again later.");
      console.error("❌ Error fetching recipes:", err);
    } finally {
      setLoading(false);
    }
  };
  
  

  const filterRecipesByDuration = (duration: string) => {
    if (duration === "under20") {
      setFilteredRecipes(recipes.filter(recipe => recipe.cookingTime && recipe.cookingTime <= 20));
    } else if (duration === "above20") {
      setFilteredRecipes(recipes.filter(recipe => recipe.cookingTime && recipe.cookingTime > 20));
    } else {
      setFilteredRecipes(recipes); // Reset the filter
    }
  };

  const filterRecipesByCuisine = (cuisine: string) => {
    setSelectedCuisine(cuisine);
    filterRecipesByCuisineAndDishTypes(selectedDishTypes, cuisine);
  };

  const filterRecipesByCuisineAndDishTypes = (selectedDishTypes: string[], selectedCuisine: string) => {
    setFilteredRecipes(
      recipes.filter((recipe) => {
        const matchesDishType = selectedDishTypes.length === 0 || recipe.dishTypes.some((type: string) => selectedDishTypes.includes(type));
        const matchesCuisine = !selectedCuisine || recipe.cuisines?.includes(selectedCuisine);

        return matchesDishType && matchesCuisine;
      })
    );
  };

  const filterRecipesByDishTypes = (selectedDishTypes: string[]) => {
    setSelectedDishTypes(selectedDishTypes);
    filterRecipesByCuisineAndDishTypes(selectedDishTypes, selectedCuisine);
  };




  const openRecipeInNewTab = async (recipe: any) => {
    try {
      const response = await axios.get(
        `https://api.spoonacular.com/recipes/${recipe.id}/information`,
        {
          params: { apiKey: API_KEY },
        }
      );

      const nutritionLabel = await axios.get(
        `https://api.spoonacular.com/recipes/${recipe.id}/nutritionLabel`,
        {
          params: { apiKey: API_KEY },
        }
      );

      const cardResponse = await axios.get(
        `https://api.spoonacular.com/recipes/${recipe.id}/card`,
        {
          params: { apiKey: API_KEY },
        }
      );

      const newTab = window.open("", "_blank");
      if (newTab) {
        newTab.document.write(`
          <html>
            <head>
              <title>${response.data.title}</title>
              <style>
                body {
                  font-family: Arial, sans-serif;
                  padding: 30px;
                  margin: 0;
                  background-color: #f5f5f5;
                  color: #333;
                }
                h1 {
                  font-size: 2.5rem;
                  margin-bottom: 30px;
                  color: #047857;
                  text-align: center;
                }
                .container {
                  display: flex;
                  justify-content: space-between;
                  gap: 30px;
                  max-width: 1200px;
                  margin: 0 auto;
                }
                .nutrition {
                  width: 320px;
                  padding: 25px;
                  background-color: #ffffff;
                  border-radius: 10px;
                  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                }
                .nutrition h2 {
                  font-size: 1.75rem;
                  color: #555;
                  margin-bottom: 20px;
                }
                .nutrition-widget {
                  font-size: 1rem;
                  color: #666;
                }
                .recipe-details {
                  flex: 1;
                  background-color: #ffffff;
                  padding: 25px;
                  border-radius: 10px;
                  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                }
                .recipe-details img {
                  max-width: 100%;
                  height: auto;
                  margin-bottom: 25px;
                  border-radius: 10px;
                  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .recipe-card {
                  max-width: 100%;
                  margin-top: 20px;
                  border-radius: 10px;
                }
                .recipe-details h2 {
                  font-size: 1.75rem;
                  color: #555;
                  margin-bottom: 15px;
                }
                ul {
                  list-style-type: none;
                  padding: 0;
                  margin-bottom: 20px;
                }
                li {
                  font-size: 1rem;
                  color: #666;
                  margin-bottom: 12px;
                  line-height: 1.5;
                }
                p {
                  font-size: 1.125rem;
                  color: #555;
                  line-height: 1.6;
                }
                .recipe-details ul {
                  margin-top: 20px;
                }
                .button {
    margin-top: 20px;
    padding: 10px 20px;
    background-color: #047857;
    color: white;
    border: none;  /* Remove border for consistency */
    border-radius: 5px;
    text-decoration: none;
    font-size: 1rem;
    display: inline-block;
    transition: background-color 0.3s;
    cursor: pointer;  /* Ensure buttons have pointer cursor */
}

.button:hover {
    background-color: #065f46;
}

.button-container {
    margin-top: 20px;
    display: flex;
    gap: 10px;
    justify-content: center;
}

               
              </style>
            </head>
            <body>
              <h1>${response.data.title}</h1>
              <div class="container">
                <div class="nutrition">
                  <div class="nutrition-widget">
                    ${nutritionLabel.data}
                  </div>
                </div>
                <div class="recipe-details">
                  <center><img src="${response.data.image}" alt="${response.data.title}" /></center>
                  <h2>Ingredients:</h2>
                  <ul>
                    ${response.data.extendedIngredients.map((i: any) => `<li>${i.original}</li>`).join("")}
                  </ul>
                  <h2>Instructions:</h2>
                  <p>${response.data.instructions}</p>
                  <h2>Recipe Card:</h2>
                  <img src="${cardResponse.data.url}" alt="Recipe Card" class="recipe-card"/>
                  <div class="button-container">
                <a href="javascript:window.close();" class="button">Close</a>
                <button id="save-recipe" class="button">Save Recipe</button>
                <button id="leftover-food" class="button">Leftover Food</button>
              </div>
              <div id="leftover-recipe" class="leftover-recipe" style="display: none;"></div>
            </div>
          </div>

              <script>
                document.getElementById('save-recipe').addEventListener('click', function() {
                  const recipeData = ${JSON.stringify(response.data)};
                  // window.saveRecipe(recipeData);
                  window.opener.saveRecipe(recipeData);
                  alert('Recipe saved successfully!');
                });

              document.getElementById('leftover-food').addEventListener('click', async function() {
  const leftoverContainer = document.getElementById('leftover-recipe');
  leftoverContainer.innerHTML = "<p>Generating leftover recipe...</p>";
  leftoverContainer.style.display = "block";

  try {
    const geminiResponse = await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyBxsKblg9RCQB742H-bNE_QGuEi1amM_aY", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: "I made '${response.data.title}' with these ingredients: ${response.data.extendedIngredients.map(i => i.original).join(', ')}. Instructions: ${response.data.instructions}. Now '${response.data.title}' is remaining. Can you suggest a new dish ?  by modifying '${response.data.title}' with detailed steps? becuase i dont want to waste the dish in the bin"
          }]
        }]
      })
    });

    const geminiData = await geminiResponse.json();
    leftoverContainer.innerHTML = "<h2>Leftover Recipe:</h2><p>" + (geminiData.candidates?.[0]?.content?.parts?.[0]?.text || "Could not generate a recipe.") + "</p>";
  } catch (error) {
    leftoverContainer.innerHTML = "<p>Failed to generate leftover recipe. Please try again.</p>";
  }
});
</script>


            </body>
          </html>
        `);
        newTab.document.close();
      }
    } catch (err) {
      console.error("Failed to open recipe in a new tab.", err);
    }
  };
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar username={user?.fullName} onLogout={() => setUser(null)} savedRecipesCount={savedRecipes.length} />
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Welcome to Food Muse!</h1>
          <p className="text-lg text-gray-600">Enter your ingredients and let's create something delicious</p>
        </div>
  
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold mb-4">What's in your kitchen?</h2>
          <IngredientInput onGenerateRecipes={fetchRecipes} />
          <FilterCard
            onFilterDuration={filterRecipesByDuration}
            onFilterDishTypes={filterRecipesByDishTypes}
            onFilterCuisine={filterRecipesByCuisine} // Pass cuisine filter handler
          />
  
          {loading && <p className="text-emerald-600">Loading recipes...</p>}
          {error && <p className="text-red-600">{error}</p>}
  
          <div className="mt-6">
          <h3 className="text-2xl font-semibold mb-4">Top Healthiest Recipe Based On Your Profile</h3>

            {/* Combine Healthiest Recipe and Recommended Recipes into one section */}
            {filteredRecipes.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Display Top 1 Healthiest Recipe First */}
                <div className="bg-green-100 border border-green-400 rounded-lg shadow-md hover:shadow-xl transition-shadow p-4 cursor-pointer">
                  {[...filteredRecipes]
                    .sort((a, b) => b.healthScore - a.healthScore)
                    .slice(0, 1)
                    .map((recipe) => (
                      <div
                      key={recipe.id}
                      onClick={() => openRecipeInNewTab(recipe)}
                    >
                        <img
                          src={recipe.image}
                          alt={recipe.title}
                          className="w-full h-40 object-cover rounded-md mb-4"
                        />
                        <h4 className="font-semibold text-lg mb-2 text-green-800">{recipe.title}</h4>
                        
                      </div>
                    ))}
                </div>
  
                {/* Recommended Recipes */}
                {recommendedRecipes.length > 0 && recommendedRecipes.map((recipe) => (
                  <div
                    key={recipe.id}
                    className="bg-green-100 border border-green-400 rounded-lg shadow-md hover:shadow-xl transition-shadow p-4 cursor-pointer"
                    onClick={() => openRecipeInNewTab(recipe)}
                  >
                    <img
                      src={recipe.image}
                      alt={recipe.title}
                      className="w-full h-40 object-cover rounded-md mb-4"
                    />
                    <h4 className="font-semibold text-lg mb-2">{recipe.title}</h4>
                  </div>
                ))}
              </div>
            )}
          </div>
  
          {/* All Recipes */}
          {filteredRecipes.length > 0 && (
            <div>
              <h3 className="text-2xl font-semibold mt-8 mb-4">All Recipes:</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredRecipes.map((recipe) => (
                  <div
                    key={recipe.id}
                    className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow p-4 cursor-pointer"
                    onClick={() => openRecipeInNewTab(recipe)} // ✅ Now consistent for all recipes
                  >
                    <img
                      src={recipe.image}
                      alt={recipe.title}
                      className="w-full h-40 object-cover rounded-md mb-4"
                    />
                    <h4 className="font-semibold text-lg mb-2">{recipe.title}</h4>
                    <p className="text-sm text-gray-600 mb-2">
                      Used ingredients: {recipe.usedIngredients.map((i: any) => i.name).join(", ")}
                    </p>
                    <p className="text-sm text-gray-600">
                      Missed ingredients: {recipe.missedIngredients.map((i: any) => i.name).join(", ")}
                    </p>
                    <p className="text-sm text-gray-600">
                      Cooking time: {recipe.cookingTime ? `${recipe.cookingTime} minutes` : "Not available"}
                    </p>
                    <div className="mt-4 text-sm text-gray-600">
                      <p>Calories: {recipe.nutrition.calories}</p>
                      <p>Carbs: {recipe.nutrition.carbs}</p>
                      <p>Protein: {recipe.nutrition.protein}</p>
                      <p>Healthscore: {recipe.healthScore}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}  
function App() {
  return (
    <Router>
      <UserProvider>
        <Routes>
          <Route path="/" element={<AppContent />} />
          <Route path="/saved-recipes" element={<RecipeComponent />} />
        </Routes>
      </UserProvider>
    </Router>
  );
}

export default App;