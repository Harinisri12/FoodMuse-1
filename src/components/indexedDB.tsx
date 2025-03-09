import { openDB } from 'idb';

const DB_NAME = 'RecipeDB';
const STORE_NAME = 'savedRecipes';

async function getDB() {
    return openDB(DB_NAME, 1, {
        upgrade(db) {
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: 'id' }); // 'id' as the primary key
            }
        },
    });
}

export async function saveRecipeToDB(recipe: any) {
    const db = await getDB();
    await db.put(STORE_NAME, recipe);
}

export async function getSavedRecipesFromDB() {
    const db = await getDB();
    return await db.getAll(STORE_NAME);
}

export async function deleteRecipeFromDB(recipeId: any) {
    const db = await getDB();
    await db.delete(STORE_NAME, recipeId);
}