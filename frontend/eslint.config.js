import js from '@eslint/js';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true }
      },
      globals: {
        ...globals.browser
      }
    },
    plugins: {
      react,
      'react-hooks': reactHooks
    },
    settings: {
      react: { version: 'detect' }
    },
    rules: {
      ...react.configs.flat.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // Vite injects React automatically and this codebase does not use
      // prop-types; keep the signal-to-noise ratio high.
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
      'no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
      // The classic hook rules are enforced as errors.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'error',
      // The React-Compiler readiness rules from eslint-plugin-react-hooks v7
      // flag long-standing intentional patterns in this app (resetting locate
      // UI state when leaving /locate, deriving the upload preview object URL
      // in an effect, and driving an HTMLAudioElement through a ref).
      // Rewriting those would change runtime behavior, so they stay off.
      'react-hooks/set-state-in-effect': 'off',
      'react-hooks/immutability': 'off'
    }
  },
  {
    files: ['vite.config.js', 'eslint.config.js'],
    languageOptions: {
      globals: {
        ...globals.node
      }
    }
  }
];
