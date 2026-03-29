import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders credit risk header and analysis button", () => {
  render(<App />);
  expect(screen.getByText(/Credit Risk/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Analyser/i })).toBeInTheDocument();
});
