import { configureStore } from '@reduxjs/toolkit';

export const mockStore = configureStore({
  reducer: {
    // Add your reducers here
  },
});

export type RootState = ReturnType<typeof mockStore.getState>;
export type AppDispatch = typeof mockStore.dispatch;
