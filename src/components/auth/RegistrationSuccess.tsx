import React from 'react';
import { CheckCircle } from 'lucide-react';

interface RegistrationSuccessProps {
  onContinueToLogin: () => void;
}

export function RegistrationSuccess({ onContinueToLogin }: RegistrationSuccessProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full text-center space-y-8">
        <div className="flex flex-col items-center">
          <CheckCircle className="w-16 h-16 text-emerald-500 mb-4" />
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Registration Successful!</h2>
          <p className="text-gray-600 mb-8">Your account has been created successfully.</p>
          <button
            onClick={onContinueToLogin}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
          >
            Continue to Login
          </button>
        </div>
      </div>
    </div>
  );
}