import { render as rtlRender, RenderOptions } from '@testing-library/react';
import { ReactElement, ReactNode } from 'react';
import { Provider } from 'react-redux';
import { MemoryRouter } from 'react-router-dom';
import { mockStore } from './mockStore';

interface ExtendedRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  route?: string;
  path?: string;
}

export const renderWithProviders = (
  ui: ReactElement,
  {
    store = mockStore,
    ...renderOptions
  }: { store?: typeof mockStore } = {}
) => {
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <Provider store={store}>
      <MemoryRouter initialEntries={['/']}>
        {children}
      </MemoryRouter>
    </Provider>
  );

  return rtlRender(ui, { wrapper: Wrapper, ...renderOptions });
};

export { renderWithProviders };
