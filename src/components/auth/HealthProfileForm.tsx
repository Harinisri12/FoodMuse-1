import React from 'react';
import { Heart, Scale, Ruler } from 'lucide-react';
import type { HealthProfile } from '../../types';

interface HealthProfileFormProps {
  healthProfile: HealthProfile;
  onChange: (profile: HealthProfile) => void;
  onBack: () => void;
  onSubmit: () => void;
}

export function HealthProfileForm({ healthProfile, onChange, onBack, onSubmit }: HealthProfileFormProps) {
  const handleChange = (field: keyof HealthProfile, value: any) => {
    onChange({ ...healthProfile, [field]: value });
  };

  const handleHealthConditionChange = (condition: keyof HealthProfile['healthConditions'], value: boolean) => {
    onChange({
      ...healthProfile,
      healthConditions: {
        ...healthProfile.healthConditions,
        [condition]: value
      }
    });
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <Heart className="mx-auto h-12 w-12 text-emerald-500" />
        <h2 className="mt-2 text-2xl font-bold">Health Profile</h2>
        <p className="text-gray-600">Help us personalize your recipe recommendations</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Height (cm)</label>
          <div className="mt-1 relative rounded-md shadow-sm">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
              <Ruler className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="number"
              value={healthProfile.height}
              onChange={(e) => handleChange('height', Number(e.target.value))}
              className="pl-10 block w-full rounded-md border-gray-300 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Weight (kg)</label>
          <div className="mt-1 relative rounded-md shadow-sm">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center">
              <Scale className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="number"
              value={healthProfile.weight}
              onChange={(e) => handleChange('weight', Number(e.target.value))}
              className="pl-10 block w-full rounded-md border-gray-300 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Health Conditions</label>
        <div className="space-y-2">
          {Object.entries(healthProfile.healthConditions).map(([condition, value]) => (
            condition !== 'allergies' && (
              <div key={condition} className="flex items-center">
                <input
                  type="checkbox"
                  checked={value}
                  onChange={(e) => handleHealthConditionChange(condition as keyof HealthProfile['healthConditions'], e.target.checked)}
                  className="h-4 w-4 text-emerald-600 focus:ring-emerald-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-700">
                  {condition.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </label>
              </div>
            )
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Allergies</label>
        <input
          type="text"
          placeholder="Enter allergies separated by commas"
          value={healthProfile.healthConditions.allergies.join(', ')}
          onChange={(e) => handleHealthConditionChange('allergies', e.target.value.split(',').map(s => s.trim()))}
          className="mt-1 block w-full rounded-md border-gray-300 focus:ring-emerald-500 focus:border-emerald-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Fitness Goal</label>
        <select
          value={healthProfile.fitnessGoal}
          onChange={(e) => handleChange('fitnessGoal', e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 focus:ring-emerald-500 focus:border-emerald-500"
        >
          <option value="weightLoss">Weight Loss</option>
          <option value="maintenance">Maintenance</option>
          <option value="muscleGain">Muscle Gain</option>
        </select>
      </div>

      <div className="flex gap-4">
        <button
          type="button"
          onClick={onBack}
          className="flex-1 py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
        >
          Back
        </button>
        <button
          type="button"
          onClick={onSubmit}
          className="flex-1 py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
        >
          Complete Registration
        </button>
      </div>
    </div>
  );
}