import React from 'react';
import { Lock, Mail, User } from 'lucide-react';
import { FormInput } from '../common/FormInput';
import type { RegistrationFormData } from '../../types';

interface RegistrationFormProps {
  formData: RegistrationFormData;
  errors: { field: string; message: string; }[];
  onInputChange: (field: keyof RegistrationFormData, value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onSwitchToLogin: () => void;
}

export function RegistrationForm({ 
  formData, 
  errors, 
  onInputChange, 
  onSubmit,
  onSwitchToLogin 
}: RegistrationFormProps) {
  const getError = (field: string) => errors.find(error => error.field === field)?.message;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">Create your account</h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={onSubmit}>
          <div className="rounded-md shadow-sm space-y-4">
            <FormInput
              id="fullName"
              type="text"
              icon={<User className="h-5 w-5 text-gray-400" />}
              value={formData.fullName}
              onChange={(e) => onInputChange('fullName', e.target.value)}
              placeholder="Full Name"
              error={getError('fullName')}
            />

            <FormInput
              id="username"
              type="text"
              icon={<User className="h-5 w-5 text-gray-400" />}
              value={formData.username}
              onChange={(e) => onInputChange('username', e.target.value)}
              placeholder="Username"
              error={getError('username')}
            />

            <FormInput
              id="email"
              type="email"
              icon={<Mail className="h-5 w-5 text-gray-400" />}
              value={formData.email}
              onChange={(e) => onInputChange('email', e.target.value)}
              placeholder="Email address"
              error={getError('email')}
            />

            <FormInput
              id="password"
              type="password"
              icon={<Lock className="h-5 w-5 text-gray-400" />}
              value={formData.password}
              onChange={(e) => onInputChange('password', e.target.value)}
              placeholder="Password"
              error={getError('password')}
            />
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
            >
              Create account
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={onSwitchToLogin}
              className="text-sm text-emerald-600 hover:text-emerald-500"
            >
              Already have an account? Sign in
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}