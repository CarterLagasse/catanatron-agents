import { test, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

test('renders opponent choice and settings on the home page', () => {
  const { getByText, getByRole, getAllByRole } = render(<App />);
  expect(getByText(/Choose your opponent/i)).toBeInTheDocument();
  expect(getByText(/Points to win/i)).toBeInTheDocument();
  expect(getByText(/Card discard limit/i)).toBeInTheDocument();
  expect(getByText(/Seats/i)).toBeInTheDocument();
  expect(getAllByRole('button', { name: /Catanatron/i }).length).toBeGreaterThan(0);
  expect(getByRole('button', { name: /^Play/i })).toBeInTheDocument();
  expect(getByRole('link', { name: 'Classic' }).getAttribute('href')).toMatch(/:4001$/);
});
