// ESM-загрузчик для тестов: разрешает относительные импорты без расширения
// (как делает Vite в исходниках src/), добавляя ".js".
export async function resolve(specifier, context, next) {
  if (specifier.startsWith('.') && !specifier.endsWith('.js') && !specifier.endsWith('.mjs')) {
    try { return await next(specifier + '.js', context); } catch {}
  }
  return next(specifier, context);
}
