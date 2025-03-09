export interface HealthProfile {
  height: number; // in cm
  weight: number; // in kg
  healthConditions: {
    diabetes: boolean;
    highBloodPressure: boolean;
    heartDisease: boolean;
    allergies: string[];
  };
  dietaryRestrictions: string[];
  fitnessGoal: 'weightLoss' | 'maintenance' | 'muscleGain';
}

export interface User {
  username: string;
  email: string;
  fullName: string;
  healthProfile: HealthProfile;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface RegistrationFormData {
  username: string;
  email: string;
  password: string;
  fullName: string;
  healthProfile: HealthProfile;
}